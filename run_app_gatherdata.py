from DataGather.gatherdata_gui import windowClass
import wx

if __name__ == '__main__':

    app = wx.App()
    windowClass(None)

    app.MainLoop()