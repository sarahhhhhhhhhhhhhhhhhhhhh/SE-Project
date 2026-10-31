Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = dir
If Not fso.FileExists(dir & "\venv\Scripts\pythonw.exe") Then
  MsgBox "Not installed yet. Run Install.bat first.", 16, "Resume Ranker"
Else
  sh.Run """" & dir & "\venv\Scripts\pythonw.exe"" """ & dir & "\app\desktop.py""", 0, False
End If
