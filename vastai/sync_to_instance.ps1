# 把本地仓库改动同步到 Vast.ai 实例（覆盖式，不删实例上已下载的模型/数据）。
#
# 用法（在 PowerShell 里）：
#   .\vastai\sync_to_instance.ps1 -Host 123.45.67.89 -Port 40000
#
# 可选参数：
#   -User       SSH 用户名，默认 root
#   -RepoDir    本地仓库路径，默认 F:\Study Files\Projects\GRPO-learning\Tree-GRPO
#   -RemoteDir  实例上的代码目录，默认 /workspace/Tree-GRPO
#
# 原理：本地 tar 打包（排除大目录）-> scp 上传 -> 实例上 tar 解压覆盖。
# 只覆盖同名文件、不删除远端多余文件（本地已删的文件远端不会自动消失，如需精确删除用 rsync）。

param(
    [Parameter(Mandatory = $true)][string]$Host,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$User = "root",
    [string]$RepoDir = "F:\Study Files\Projects\GRPO-learning\Tree-GRPO",
    [string]$RemoteDir = "/workspace/Tree-GRPO"
)

$ErrorActionPreference = "Stop"
$tmp = Join-Path $env:TEMP ("treegrpo_sync_" + [guid]::NewGuid().ToString("N") + ".tgz")

Write-Host "==> 1/3 打包仓库（排除 .git / models / data / retrieval / checkpoint 等大目录）..."
Push-Location $RepoDir
& tar -czf $tmp `
    --exclude='./.git' `
    --exclude='./models' `
    --exclude='./data' `
    --exclude='./retrieval' `
    --exclude='./verl_checkpoints' `
    --exclude='./verl_log' `
    --exclude='./sft_checkpoint' `
    .
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "tar 打包失败" }
Pop-Location

Write-Host "==> 2/3 上传到 ${User}@${Host}:${Port} ..."
& scp -P $Port $tmp "${User}@${Host}:/tmp/repo_sync.tgz"
if ($LASTEXITCODE -ne 0) { Remove-Item $tmp -ErrorAction SilentlyContinue; throw "scp 上传失败" }

Write-Host "==> 3/3 在实例上解压到 ${RemoteDir} ..."
& ssh -p $Port "${User}@${Host}" "tar -xzf /tmp/repo_sync.tgz -C ${RemoteDir} && rm -f /tmp/repo_sync.tgz"
if ($LASTEXITCODE -ne 0) { Remove-Item $tmp -ErrorAction SilentlyContinue; throw "ssh 解压失败" }

Remove-Item $tmp -ErrorAction SilentlyContinue
Write-Host "完成。代码已同步到实例，重新运行对应命令即可（.sh 用 bash 调用，.py 下次运行自动生效）。"
