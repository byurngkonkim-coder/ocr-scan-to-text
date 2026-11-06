param([string]$Action = 'check')
# Used by START_OCR.bat / STOP_OCR.bat to find the OCR server on port 7860.
# netstat is not reliable here: a running server's socket showed FIN_WAIT_1 or was not listed at all
# while it still answered (2026-11-06). So: try to connect, and find the server by its command line.
#   check : exit 0 = port free, 2 = this OCR server is running, 3 = another program uses the port
#   up    : exit 0 = port answers, 1 = not yet
#   stop  : stop every ocr_gradio.py process; exit 0 = stopped, 1 = none found
$ErrorActionPreference = 'SilentlyContinue'

function Test-Up {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', 7860); return $true } catch { return $false } finally { $c.Close() }
}

function Get-Server {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'ocr_gradio\.py' }
}

switch ($Action) {
    'up' { if (Test-Up) { exit 0 } else { exit 1 } }
    'check' {
        if (-not (Test-Up)) { exit 0 }
        if (Get-Server) { exit 2 } else { exit 3 }
    }
    'stop' {
        $p = @(Get-Server)
        if ($p.Count -eq 0) { exit 1 }
        $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        exit 0
    }
}
exit 1
