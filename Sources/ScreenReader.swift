import AppKit
import ScreenCaptureKit
import Vision

enum Recognition {
    static func lines(using handler: VNImageRequestHandler) throws -> [String] {
        let request = VNRecognizeTextRequest()
        // .fast 认不出中文（本机 macOS 27.2 / Vision rev3 实测整行返回空），只能用于英文。
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
        request.automaticallyDetectsLanguage = true
        try handler.perform([request])
        return (request.results ?? []).compactMap { result in
            guard let match = result.topCandidates(1).first, match.confidence >= 0.45 else { return nil }
            return match.string
        }
    }
}

// 打字时输入框里的拼音和光标每按一键都在动，整帧哈希每秒都不同，所以按"脏了多少格"判断：
// 一行文字只脏一两格，翻页、切窗口、滚动会脏一大片。
enum FrameDiff {
    static let grid = 16

    // 给 16x16 网格的每一格算一个采样哈希，只保留 256 个 UInt64，不复制整帧像素。
    static func hashes(_ buffer: CVPixelBuffer) -> [UInt64]? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard width >= grid, height >= grid else { return nil }
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let columns = width / grid, rows = height / grid
        var result = [UInt64](repeating: 0, count: grid * grid)
        for row in 0..<grid {
            for column in 0..<grid {
                var hash: UInt64 = 0xcbf2_9ce4_8422_2325
                for y in Swift.stride(from: row * rows, to: (row + 1) * rows, by: 4) {
                    for x in Swift.stride(from: column * columns, to: (column + 1) * columns, by: 4) {
                        let offset = y * stride + x * 4
                        for channel in 0..<3 {
                            hash = (hash ^ UInt64(pixels[offset + channel])) &* 0x100_0000_01b3
                        }
                    }
                }
                result[row * grid + column] = hash
            }
        }
        return result
    }

    static func dirtyRatio(_ current: [UInt64], _ previous: [UInt64]) -> Double {
        let dirty = zip(current, previous).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
        return Double(dirty) / Double(current.count)
    }
}

// Each capture session owns its queue and tokenizer. Drop results from stopped sessions in the UI.
private final class DisplayReader: NSObject, SCStreamOutput, SCStreamDelegate {
    private let queue = DispatchQueue(label: "RimeScreenWords.recognition", qos: .utility)
    private let index = WordIndex()
    private var lastLines: [String]?
    // 打字时输入框里的拼音、光标每按一键都在动，整帧哈希每秒都不同，所以按"脏了多少格"判断。
    // 一行文字只脏 1-2 格，翻页、切窗口、滚动会脏一大片。
    private var lastTiles: [UInt64]?
    private var skippedFrames = 0
    private static let dirtyThreshold = 0.05
    private static let maxSkippedFrames = 8
    private var stream: SCStream?
    private var starting = false
    let update: ([Entry]) -> Void
    let failure: (Error) -> Void

    init(update: @escaping ([Entry]) -> Void, failure: @escaping (Error) -> Void) {
        self.update = update
        self.failure = failure
    }

    @MainActor func start(filter: SCContentFilter) async throws {
        let config = SCStreamConfiguration()
        // 采样宽度上限从 3072 降到 1800：Retina 2x 下约等于 0.6 倍分辨率，OCR 像素量减半以上
        let scale = min(CGFloat(filter.pointPixelScale), 1800 / max(1, filter.contentRect.width))
        config.width = max(1, Int(filter.contentRect.width * scale))
        config.height = max(1, Int(filter.contentRect.height * scale))
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.queueDepth = 3
        config.showsCursor = false
        config.capturesAudio = false
        let capture = SCStream(filter: filter, configuration: config, delegate: self)
        try capture.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        stream = capture
        starting = true
        defer { starting = false }
        try await capture.startCapture()
        // Commit or Escape may arrive before ScreenCaptureKit finishes starting.
        if stream == nil { try? await capture.stopCapture() }
    }

    @MainActor func stop() async {
        let capture = stream
        stream = nil
        if !starting { try? await capture?.stopCapture() }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async { self.failure(error) }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sample: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sample.isValid,
              let data = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false)
                as? [[SCStreamFrameInfo: Any]],
              let raw = data.first?[.status] as? Int, let status = SCFrameStatus(rawValue: raw)
        else { return }
        if status == .idle || status == .started { return }
        do {
            let lines: [String]
            if status == .complete, let buffer = sample.imageBuffer {
                // 只有输入框那一两行在变时跳过识别，翻页、切窗口这类大变化照常识别。
                let tiles = FrameDiff.hashes(buffer)
                let previous = lastTiles
                lastTiles = tiles
                if let tiles, let previous, previous.count == tiles.count,
                   FrameDiff.dirtyRatio(tiles, previous) < Self.dirtyThreshold,
                   skippedFrames < Self.maxSkippedFrames {
                    skippedFrames += 1
                    return
                }
                skippedFrames = 0
                lines = try Recognition.lines(using: VNImageRequestHandler(cvPixelBuffer: buffer))
            } else {
                lines = []
            }
            guard lines != lastLines else { return }
            lastLines = lines
            let entries = index.entries(lines: lines)
            DispatchQueue.main.async { self.update(entries) }
        } catch {
            DispatchQueue.main.async { self.failure(error) }
        }
    }
}

// Keep each display's most recent result until that display changes or leaves.
@MainActor
final class ScreenReader {
    private struct Capture {
        let id: UUID
        let reader: DisplayReader
        let size: CGSize
        let scale: Float
    }

    private let update: ([Entry]) -> Void
    private let status: (Int, Error?) -> Void
    private var captures: [CGDirectDisplayID: Capture] = [:]
    private var words: [CGDirectDisplayID: [Entry]] = [:]
    private var errors: [CGDirectDisplayID: Error] = [:]
    private var refreshing = false
    private var needsRefresh = false
    private var stopped = false

    init(update: @escaping ([Entry]) -> Void, status: @escaping (Int, Error?) -> Void) {
        self.update = update
        self.status = status
    }

    // Coalesce notifications received while ScreenCaptureKit is enumerating or starting.
    func refresh(retryFailed: Bool = true) {
        guard !stopped else { return }
        if retryFailed { errors.removeAll() }
        needsRefresh = true
        guard !refreshing else { return }
        refreshing = true
        Task {
            while needsRefresh && !stopped {
                needsRefresh = false
                do {
                    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                    if stopped || needsRefresh { continue }
                    await reconcile(content)
                    if !stopped && !needsRefresh { status(captures.count, errors.values.first) }
                } catch {
                    if !stopped && !needsRefresh { status(captures.count, error) }
                }
            }
            refreshing = false
        }
    }

    private func reconcile(_ content: SCShareableContent) async {
        let displays = content.displays.filter { CGDisplayMirrorsDisplay($0.displayID) == kCGNullDirectDisplay }
        let connected = Set(displays.map(\.displayID))
        errors = errors.filter { connected.contains($0.key) }
        for id in Set(captures.keys).subtracting(connected) { remove(id) }
        let excluded = content.applications.filter {
            $0.bundleIdentifier == "im.rime.inputmethod.Squirrel" || $0.bundleIdentifier == Bundle.main.bundleIdentifier
        }
        for display in displays {
            guard !stopped && !needsRefresh else { return }
            let id = display.displayID
            let filter = SCContentFilter(display: display, excludingApplications: excluded, exceptingWindows: [])
            if let existing = captures[id] {
                if existing.size == filter.contentRect.size && existing.scale == filter.pointPixelScale { continue }
                remove(id)
            }
            guard errors[id] == nil else { continue }
            let token = UUID()
            let reader = DisplayReader(update: { [weak self] entries in
                guard let self, self.captures[id]?.id == token else { return }
                self.words[id] = entries
                self.publish()
            }, failure: { [weak self] error in
                guard let self, self.captures[id]?.id == token else { return }
                self.errors[id] = error
                self.remove(id)
                self.refresh(retryFailed: false)
            })
            captures[id] = Capture(id: token, reader: reader, size: filter.contentRect.size, scale: filter.pointPixelScale)
            do { try await reader.start(filter: filter) }
            catch {
                if captures[id]?.id == token {
                    errors[id] = error
                    remove(id)
                }
            }
        }
    }

    private func publish() { update(words.values.flatMap { $0 }) }

    private func remove(_ id: CGDirectDisplayID) {
        guard let capture = captures.removeValue(forKey: id) else { return }
        if words.removeValue(forKey: id) != nil && !stopped { publish() }
        Task { await capture.reader.stop() }
    }

    func stop() {
        stopped = true
        for id in Array(captures.keys) { remove(id) }
    }
}
