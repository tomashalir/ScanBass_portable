; ScanBass_exe.iss — installer for packaged EXE (no Python needed, no console)
#define MyAppName "ScanBass"
#define MyAppVersion "0.9.0"
#define MyAppPublisher "ScanBass"
#define MyAppId "{{A1E7F3A0-8C8E-4D92-9B6E-0C9E7F50B812}}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={userappdata}\ScanBass
DefaultGroupName=ScanBass
DisableStartupPrompt=yes
DisableDirPage=yes
DisableReadyMemo=yes
DisableFinishedPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2/ultra
SolidCompression=yes
OutputBaseFilename=ScanBassSetup
WizardStyle=modern
SetupIconFile={#SourcePath}\..\ScanBass_portable\ScanBass_roll_mint.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\ScanBass\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\ScanBass"; Filename: "{app}\ScanBass.exe"; WorkingDir: "{app}"; IconFilename: "{app}\ScanBass_roll_mint.ico"
Name: "{userdesktop}\ScanBass"; Filename: "{app}\ScanBass.exe"; WorkingDir: "{app}"; IconFilename: "{app}\ScanBass_roll_mint.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\ScanBass.exe"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\__pycache__"