# Rogator v2.1.0 - GitHub 一键发布脚本
# 用法: powershell -ExecutionPolicy Bypass -File .\publish.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "`n=== Rogator GitHub 发布 ===`n" -ForegroundColor Cyan
if (-not (Test-Path ".git")) {
Write-Host "[1/4] 初始化 Git 仓库..." -ForegroundColor Yellow
git init
git branch -M main
} else {
Write-Host "[1/4] Git 仓库已存在，跳过" -ForegroundColor Green
}
Write-Host "[2/4] 暂存文件并提交..." -ForegroundColor Yellow
git add .
git status --short
git commit -m "Initial commit: Rogator v2.1.0 - Qwen AI adapter server" 2>&1
Write-Host "[3/4] 创建 GitHub 仓库并推送..." -ForegroundColor Yellow
$existing = gh repo view --json url -q .url 2>$null
if ($existing) {
Write-Host "  仓库已存在: $existing" -ForegroundColor Yellow
$remotes = git remote
if (-not ($remotes -contains "origin")) {
git remote add origin $existing
}
git push -u origin main --force-with-lease 2>$null
if ($LASTEXITCODE -ne 0) { git push -u origin main }
} else {
gh repo create rogator --public --source=. --remote=origin --push
}
Write-Host "`n[4/4] 验证结果..." -ForegroundColor Yellow
gh repo view --json name,url,isPrivate,createdAt,defaultBranchRef
Write-Host "`n=== 发布完成！ ===`n" -ForegroundColor Green