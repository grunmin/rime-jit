import AppKit
import Carbon
import ServiceManagement

@main
@MainActor
final class App: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private var status: NSStatusItem!
    private let toggle = NSMenuItem()
    private let loginItem = NSMenuItem(title: "登录时启动", action: #selector(toggleLoginItem), keyEquivalent: "")
    private let capturePreference = "captureEnabled"
    private var cache: Snapshot?
    private var monitor: CompositionMonitor?
    private var reader: ScreenReader?
    private var session = UUID()
    private var timer: Timer?
    private var captureStatus = ""
    // 锁屏、休眠、切换用户只是临时挂起，不能清掉取词开关；解除后自动恢复。
    private var suspensions = Set<String>()
    // 连续失败达到上限才放弃并弹窗，睡眠唤醒后的一两次流中断属于正常现象。
    private var captureFailures = 0
    private static let maxCaptureFailures = 3
    private var paused: Bool { !suspensions.isEmpty }

    static func main() {
        let app = NSApplication.shared
        let delegate = App()
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        status = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let menu = NSMenu()
        toggle.target = self
        toggle.action = #selector(toggleCapture)
        menu.addItem(toggle)
        menu.delegate = self
        loginItem.target = self
        menu.addItem(loginItem)
        menu.addItem(.separator())
        let quit = menu.addItem(withTitle: "退出屏幕候选词", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quit.target = NSApp
        status.menu = menu
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/RimeScreenWords")
        do { cache = try Snapshot(directory: directory) }
        catch { show(error) }
        NotificationCenter.default.addObserver(self, selector: #selector(screensChanged),
                                               name: NSApplication.didChangeScreenParametersNotification, object: nil)
        let center = NSWorkspace.shared.notificationCenter
        let pairs: [(String, Notification.Name, Notification.Name)] = [
            ("sleep", NSWorkspace.screensDidSleepNotification, NSWorkspace.screensDidWakeNotification),
            ("session", NSWorkspace.sessionDidResignActiveNotification, NSWorkspace.sessionDidBecomeActiveNotification),
        ]
        for (reason, begin, end) in pairs {
            center.addObserver(forName: begin, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.suspend(reason) }
            }
            center.addObserver(forName: end, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.resume(reason) }
            }
        }
        // 锁屏本身不会触发 sessionDidResignActive，要单独监听。
        let distributed = DistributedNotificationCenter.default()
        distributed.addObserver(forName: Notification.Name("com.apple.screenIsLocked"), object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.suspend("lock") }
        }
        distributed.addObserver(forName: Notification.Name("com.apple.screenIsUnlocked"), object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.resume("lock") }
        }
        center.addObserver(self, selector: #selector(focusChanged), name: NSWorkspace.didActivateApplicationNotification, object: nil)
        center.addObserver(self, selector: #selector(appTerminated(_:)), name: NSWorkspace.didTerminateApplicationNotification, object: nil)
        DistributedNotificationCenter.default().addObserver(self, selector: #selector(inputSourceChanged),
            name: Notification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String), object: nil)
        if UserDefaults.standard.bool(forKey: capturePreference) && CGPreflightScreenCaptureAccess() {
            armCapture()
        }
        render()
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        let state = SMAppService.mainApp.status
        loginItem.title = state == .requiresApproval ? "在系统设置中允许登录时启动…" : "登录时启动"
        loginItem.state = state == .enabled ? .on : .off
    }

    @objc private func toggleLoginItem() {
        let service = SMAppService.mainApp
        do {
            switch service.status {
            case .enabled: try service.unregister()
            case .requiresApproval: SMAppService.openSystemSettingsLoginItems()
            default:
                try service.register()
                if service.status == .requiresApproval { SMAppService.openSystemSettingsLoginItems() }
            }
        } catch {
            if service.status == .requiresApproval { SMAppService.openSystemSettingsLoginItems() }
            else { show(error) }
        }
    }

    private func render() {
        let active = reader != nil
        toggle.title = monitor != nil ? "暂停屏幕取词" : "开始屏幕取词"
        toggle.isEnabled = cache != nil
        let icon = NSImage(systemSymbolName: active ? "text.viewfinder" : "viewfinder",
                           accessibilityDescription: "屏幕候选词")
        icon?.isTemplate = true
        status.button?.image = icon
        status.button?.toolTip = active ? captureStatus
            : (monitor == nil ? "屏幕取词已暂停" : (paused ? "锁屏或休眠中，解锁后自动恢复" : "等待 Rime 输入"))
    }

    @objc private func toggleCapture() {
        if monitor != nil {
            UserDefaults.standard.set(false, forKey: capturePreference)
            stop()
            return
        }
        guard CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess() else {
            let alert = NSAlert()
            alert.messageText = "需要屏幕录制权限"
            alert.informativeText = "在系统设置中允许 RimeScreenWords 录制屏幕，然后重新打开应用并开始取词。"
            alert.addButton(withTitle: "打开系统设置")
            alert.addButton(withTitle: "取消")
            if alert.runModal() == .alertFirstButtonReturn,
               let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
                NSWorkspace.shared.open(url)
            }
            return
        }
        armCapture()
        UserDefaults.standard.set(monitor != nil, forKey: capturePreference)
    }

    private func armCapture() {
        guard let cache else { return }
        do {
            monitor = try CompositionMonitor(cacheDirectory: cache.url.deletingLastPathComponent()) { [weak self] in
                self?.compositionChanged()
            }
            compositionChanged()
            render()
        } catch { fail(error) }
    }

    private var squirrelSelected: Bool {
        let source = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
        guard let value = TISGetInputSourceProperty(source, kTISPropertyBundleID) else { return false }
        return Unmanaged<CFString>.fromOpaque(value).takeUnretainedValue() as String == "im.rime.inputmethod.Squirrel"
    }

    @objc private func focusChanged() {
        monitor?.reset()
        endComposition()
    }

    @objc private func inputSourceChanged() {
        if !squirrelSelected { focusChanged() }
    }

    @objc private func appTerminated(_ note: Notification) {
        let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
        if app?.bundleIdentifier == "im.rime.inputmethod.Squirrel" { focusChanged() }
    }

    private func suspend(_ reason: String) {
        let wasPaused = paused
        suspensions.insert(reason)
        guard !wasPaused, monitor != nil else { return }
        focusChanged()
        cache?.clear()
        render()
    }

    private func resume(_ reason: String) {
        suspensions.remove(reason)
        guard !paused else { return }
        // 唤醒后显示器可能重新枚举过，下一次输入会重新建立取词流。
        captureFailures = 0
        render()
    }

    // 安全输入（密码框、终端 Secure Keyboard Entry 等）只结束本次取词，不关开关；
    // 占用释放后下一次输入会自动恢复。
    private func secureInputActive() -> Bool {
        guard IsSecureEventInputEnabled() else { return false }
        focusChanged()
        return true
    }

    private func compositionChanged() {
        guard !paused, monitor?.isComposing == true && squirrelSelected else { endComposition(); return }
        if secureInputActive() { return }
        guard reader == nil else { return }
        let id = UUID()
        session = id
        let capture = ScreenReader(update: { [weak self] entries in
            guard let self, self.session == id else { return }
            if self.secureInputActive() { return }
            guard self.monitor?.isComposing == true else { self.endComposition(); return }
            do { try self.cache?.replace(entries) }
            catch { self.fail(error) }
        }, status: { [weak self] count, error in
            guard let self, self.session == id else { return }
            if count == 0, let error { self.captureFailed(error); return }
            if count > 0 { self.captureFailures = 0 }
            self.captureStatus = count == 0 ? "等待显示器" : "正在从 \(count) 台显示器取词"
            if let error { self.captureStatus += "。\(error.localizedDescription)" }
            self.render()
        })
        reader = capture
        captureStatus = "正在启动屏幕取词"
        render()
        do { try cache?.heartbeat() } catch { fail(error); return }
        capture.refresh()
        // Common modes keep lease expiry independent of an open menu.
        let clock = Timer(timeInterval: 1, target: self, selector: #selector(tick), userInfo: nil, repeats: true)
        RunLoop.main.add(clock, forMode: .common)
        timer = clock
    }

    @objc private func screensChanged() { reader?.refresh() }

    // 取词流中断（休眠唤醒、显示器重连等）先重试：清掉标记，下一次按键会重新发起取词。
    private func captureFailed(_ error: Error) {
        captureFailures += 1
        if captureFailures >= Self.maxCaptureFailures { captureFailures = 0; fail(error); return }
        captureStatus = "取词中断，下次输入时重试。\(error.localizedDescription)"
        focusChanged()
    }

    @objc private func tick() {
        if secureInputActive() { return }
        guard !paused, monitor?.isComposing == true && squirrelSelected else { endComposition(); return }
        do { try cache?.heartbeat() }
        catch { fail(error) }
    }

    private func endComposition() {
        guard reader != nil else { return }
        session = UUID()
        let previous = reader
        reader = nil
        timer?.invalidate()
        timer = nil
        previous?.stop()
        cache?.suspend()
        render()
    }

    @objc private func stop() {
        monitor?.stop()
        monitor = nil
        endComposition()
        cache?.clear()
        render()
    }

    private func fail(_ error: Error) {
        stop()
        show(error)
    }

    private func show(_ error: Error) {
        let alert = NSAlert(error: error)
        alert.runModal()
    }

    func applicationWillTerminate(_ notification: Notification) { stop() }
}
