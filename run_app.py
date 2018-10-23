from gui import windowClass
import wx

if __name__ == '__main__':

    app = wx.App()
    windowClass(None, title='epic window')

    app.MainLoop()