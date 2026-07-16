@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ═══════════════════════════════════════
echo   脉诊一键采集  — 青囊 PPG
echo ═══════════════════════════════════════
echo.

if "%1"=="" (
    set /p patient=请输入患者姓名: 
) else (
    set patient=%1
)

if "%patient%"=="" (
    echo 错误：患者姓名不能为空。
    pause
    exit /b 1
)

echo 患者: %patient%
echo 正在启动采集...
echo.

python pulse_diagnosis_cli.py --patient "%patient%"

echo.
if errorlevel 1 (
    echo 采集过程出现异常（exit code=%errorlevel%）
) else (
    echo 采集完成。
)

echo.
pause
