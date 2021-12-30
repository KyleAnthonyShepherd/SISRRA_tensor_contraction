import os
import sys

import win32gui
import win32con
import time


def winEnumHandler( hwnd, ctx ):
    # print(ctx)
    if win32gui.IsWindowVisible( hwnd ):
        if ctx+ ' - Adobe Acrobat Reader DC (64-bit)'==win32gui.GetWindowText( hwnd ):
            win32gui.PostMessage(hwnd,win32con.WM_CLOSE,0,0)
        print (win32gui.GetWindowText( hwnd ))

win32gui.EnumWindows( winEnumHandler, sys.argv[1] )
print(sys.argv)

# Manuscript_Dec_30.pdf - Adobe Reader

sys.exit()
