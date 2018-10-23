import wx
from main import run

class windowClass(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(windowClass, self).__init__(*args,  **kwargs)
        self.path = None
        self.basicGUI()

    def basicGUI(self):
        self.panel = wx.Panel(self)
        #wx.TextCtrl(panel, pos=(10, 10), size=(100, 100))

        self.a = wx.StaticText(self.panel, pos=(10, 120), size=(40, 20), label="Digits: ")
        self.b = wx.StaticText(self.panel, pos=(10, 150), size=(40, 20), label="Colors: ")
        self.c = wx.StaticText(self.panel, pos=(10, 180), size=(40, 20), label="Time: ")

        self.digits_text = wx.StaticText(self.panel, pos=(60, 120), size=(200, 20), label="")
        self.colors_text = wx.StaticText(self.panel, pos=(60, 150), size=(200, 20), label="")
        self.time_text = wx.StaticText(self.panel, pos=(60, 180), size=(200, 20), label="")

        self.path_text = wx.StaticText(self.panel, pos=(110, 25), size=(200, 25), label="No image selected")
        self.wait_text = wx.StaticText(self.panel, pos=(110, 65), size=(200, 25), label="No image selected")

        self.button1 = wx.Button(self.panel, pos=(10, 20), size=(90, 25), label="Choose Image")
        self.button2 = wx.Button(self.panel, pos=(10, 60), size=(90, 25), label="Detect")

        self.button1.Bind(wx.EVT_BUTTON, self.open_dialogue)
        self.button2.Bind(wx.EVT_BUTTON, self.detect)

        self.Center()
        self.SetTitle('Digit Program')
        self.Show()

    def open_dialogue(self, e):
        openFileDialog = wx.FileDialog(self.panel, "Open", "", "", "", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.ShowModal()
        self.path = openFileDialog.GetPath()
        self.path_text.SetLabel(self.path)
        openFileDialog.Destroy()

    def detect(self, e):
        if self.path is not None:
            self.wait_text.SetLabel('Processing, please wait...')
            try:
                digits, colors, time = run(self.path)
                self.set_labels(digits, colors, time)
            except:
                self.wait_text.SetLabel('Failed')
                self.set_labels("None", "None", "None")
        else:
            message = wx.MessageBox("Please select image..")

    def set_labels(self, digits, colors, time):
        self.digits_text.SetLabel(''.join(digits))
        self.colors_text.SetLabel(''.join(colors))
        self.time_text.SetLabel(str(round(time, 2))+"s")
        self.wait_text.SetLabel('Done')


