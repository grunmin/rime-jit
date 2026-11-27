import AppKit
import ScreenCaptureKit
import Vision

enum Recognition {
    static func lines(using handler: VNImageRequestHandler, request: VNRecognizeTextRequest) throws -> [String] {
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

// 每次 composition 都会新建一个 DisplayReader（新 SCStream），连续打字时每个词提交一次就是一次重建。
// 新流的第一帧没有"上一帧"可比，过去总是整屏 OCR，几十个短 composition 就排队几十次整屏 OCR，
// 打完字还要继续算十几秒。这里把上一次的画面指纹和识别结果按显示器留在进程里，
// 新会话的第一帧只要和它差异小就直接复用，不再 OCR。
private final class DisplayMemory {
    struct State {
        var tiles: [UInt64]
        var lines: [String]?
        var entries: [Entry]
        var lastOCR: Date
        var date: Date
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var states: [CGDirectDisplayID: State] = [:]
    // 超过这个时间的结果不再复用，避免久未输入后拿旧画面凑数。
    static let maxAge: TimeInterval = 60

    static func load(_ id: CGDirectDisplayID) -> State? {
        lock.lock(); defer { lock.unlock() }
        guard let state = states[id], Date().timeIntervalSince(state.date) < maxAge else { return nil }
        return state
    }

    static func save(_ id: CGDirectDisplayID, _ state: State) {
        lock.lock(); defer { lock.unlock() }
        states[id] = state
    }

    static func clear(_ id: CGDirectDisplayID) {
        lock.lock(); defer { lock.unlock() }
        states.removeValue(forKey: id)
    }
}

// 所有显示器、所有会话共用一条识别队列：任何时刻最多一个 OCR 在跑，
// 不会因为会话重建而出现多个整屏 OCR 并发抢 CPU/ANE。
private let recognitionQueue = DispatchQueue(label: "RimeScreenWords.recognition", qos: .utility)

// Each capture session owns its tokenizer. Drop results from stopped sessions in the UI.
private final class DisplayReader: NSObject, SCStreamOutput, SCStreamDelegate {
    private let queue = recognitionQueue
    private let index = WordIndex()
    private let displayID: CGDirectDisplayID
    // 已停止的会话：丢弃排队中的帧。正在跑的 OCR 不取消：会话通常比一次 OCR（1~3 秒）还短，
    // 取消等于白算，下一个会话又从零开始；让它跑完，结果写进 DisplayMemory 供下一个会话复用。
    private let stateLock = NSLock()
    private var isStopped = false
    private var lastLines: [String]?
    private var lastEntries: [Entry] = []
    // 打字时输入框里的拼音、光标每按一键都在动，整帧哈希每秒都不同，所以按"脏了多少格"判断。
    // 一行文字只脏 1-2 格，翻页、切窗口、滚动会脏一大片。
    // 阈值取 12%：2026-11-27 在 4K 屏上实测（修好像素格式后第一次真正量到），
    // 打字帧脏格 3.5%~9.4%，8% 会被击穿，打字时仍频繁整屏 OCR；翻页、切窗口实测 35% 左右，余量足够。
    private var lastTiles: [UInt64]?
    private var lastOCR = Date.distantPast
    // 跳过的安全性由时间上限兜底：距上次 OCR 超过 6 秒就重新识别一次。
    // 用时间而不是帧数：会话是一串短突发，一个会话常常只有 1 帧，按帧数计会随会话重建而失真。
    private static let dirtyThreshold = 0.12
    private static let maxSkipSeconds: TimeInterval = 6
    private var stream: SCStream?
    private var starting = false
    let update: ([Entry]) -> Void
    let failure: (Error) -> Void

    init(displayID: CGDirectDisplayID, update: @escaping ([Entry]) -> Void, failure: @escaping (Error) -> Void) {
        self.displayID = displayID
        self.update = update
        self.failure = failure
        super.init()
    }

    // 复用缓存要等到第一帧到来时再读：上一个会话的 OCR 可能还在共享队列里跑，
    // 它跑完会写入缓存，而这一帧只有排在它后面才会被处理。
    private var loadedMemory = false

    private func loadMemory() {
        guard !loadedMemory else { return }
        loadedMemory = true
        if let state = DisplayMemory.load(displayID) {
            lastTiles = state.tiles
            lastLines = state.lines
            lastEntries = state.entries
            lastOCR = state.lastOCR
        }
    }

    private var stopped: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return isStopped
    }

    private func markStopped() {
        stateLock.lock(); defer { stateLock.unlock() }
        isStopped = true
    }

    @MainActor func start(filter: SCContentFilter) async throws {
        let config = SCStreamConfiguration()
        // 采样宽度上限从 3072 降到 1800：Retina 2x 下约等于 0.6 倍分辨率，OCR 像素量减半以上
        let scale = min(CGFloat(filter.pointPixelScale), 1800 / max(1, filter.contentRect.width))
        config.width = max(1, Int(filter.contentRect.width * scale))
        config.height = max(1, Int(filter.contentRect.height * scale))
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.queueDepth = 3
        // ScreenCaptureKit 默认给 420v（YCbCr 双平面），FrameDiff 只认 BGRA，
        // 不指定的话 hashes() 永远返回 nil，脏格跳过和跨会话复用都不会生效，每一帧都整屏 OCR。
        config.pixelFormat = kCVPixelFormatType_32BGRA
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
        markStopped()
        let capture = stream
        stream = nil
        if !starting { try? await capture?.stopCapture() }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async { self.failure(error) }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sample: CMSampleBuffer, of type: SCStreamOutputType) {
        guard !stopped, type == .screen, sample.isValid,
              let data = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false)
                as? [[SCStreamFrameInfo: Any]],
              let raw = data.first?[.status] as? Int, let status = SCFrameStatus(rawValue: raw)
        else { return }
        if status == .idle || status == .started { return }
        loadMemory()
        do {
            let lines: [String]
            if status == .complete, let buffer = sample.imageBuffer {
                // 只有输入框那一两行在变时跳过识别，翻页、切窗口这类大变化照常识别。
                let tiles = FrameDiff.hashes(buffer)
                let previous = lastTiles
                lastTiles = tiles
                if let tiles, let previous, previous.count == tiles.count,
                   FrameDiff.dirtyRatio(tiles, previous) < Self.dirtyThreshold,
                   Date().timeIntervalSince(lastOCR) < Self.maxSkipSeconds {
                    // 新会话第一次命中复用：快照在上一次 composition 结束时被清掉了，要重新发布。
                    publishIfNeeded()
                    remember()
                    return
                }
                lastOCR = Date()
                lines = try Recognition.lines(using: VNImageRequestHandler(cvPixelBuffer: buffer),
                                              request: VNRecognizeTextRequest())
            } else {
                lines = []
            }
            if lines != lastLines {
                lastLines = lines
                lastEntries = index.entries(lines: lines)
                published = false
            }
            remember()
            // OCR 期间会话可能已结束：结果留在缓存里，但不再发布。
            if !stopped { publishIfNeeded() }
        } catch {
            if stopped { return }
            DispatchQueue.main.async { self.failure(error) }
        }
    }

    // 本会话是否已把 lastEntries 发给 UI（新会话的快照是空的，必须至少发一次）。
    private var published = false

    private func publishIfNeeded() {
        guard !published else { return }
        published = true
        let entries = lastEntries
        DispatchQueue.main.async { self.update(entries) }
    }

    private func remember() {
        guard let tiles = lastTiles else { return }
        DisplayMemory.save(displayID, .init(tiles: tiles, lines: lastLines, entries: lastEntries,
                                            lastOCR: lastOCR, date: Date()))
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
            let reader = DisplayReader(displayID: id, update: { [weak self] entries in
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
