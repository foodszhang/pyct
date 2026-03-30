; ============================================================
; PyCT Inno Setup 安装脚本
; 用法：ISCC.exe /DAppVersion=1.0.0 /DDistDir=... /DOutputDir=... setup.iss
; ============================================================

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

#ifndef DistDir
  #define DistDir "..\build_output\dist\PyCT"
#endif

#ifndef OutputDir
  #define OutputDir "..\build_output\installer"
#endif

[Setup]
AppName=PyCT
AppVersion={#AppVersion}
AppPublisher=Jian Zhang
DefaultDirName={autopf}\PyCT
DefaultGroupName=PyCT
OutputDir={#OutputDir}
OutputBaseFilename=PyCT_Setup_{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=
UninstallDisplayIcon={app}\PyCT.exe
DisableProgramGroupPage=yes
PrivilegesRequired=admin
WizardStyle=modern

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; 主程序（PyInstaller one-folder 输出）
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; VC++ Runtime（可选，如果你放在 scripts/redist/ 下）
; Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\PyCT"; Filename: "{app}\PyCT.exe"
Name: "{group}\卸载 PyCT"; Filename: "{uninstallexe}"
Name: "{autodesktop}\PyCT"; Filename: "{app}\PyCT.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"

[Run]
; 安装完成后可选启动
Filename: "{app}\PyCT.exe"; Description: "启动 PyCT"; Flags: nowait postinstall skipifsilent

; 静默安装 VC++ Runtime（如果你打包了的话，取消下面的注释）
; Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "安装 VC++ 运行库..."; Flags: waituntilterminated

[Code]
// 安装前检查是否已安装旧版本
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
