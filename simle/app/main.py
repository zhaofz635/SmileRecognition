import os

from simle.data.page.show_about_page import show_about_page
from simle.interface_operation.FileDownload import show_file_download_popup

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from simle.interface_design.ui_design import create_main_layout, show_contact_us_popup
from simle.interface_operation.recognize_action import open_filechooser
from simle.interface_operation.analyze_action import analyze_and_display_results

class VideoAnalyzerApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_data = None

    def build(self):
        Window.size = (800, 600) 
        Window.title = 'Simle Identify'
        layout = create_main_layout(self)
        self.recognize_button = layout.ids['recognize_button']
        self.analyze_button = layout.ids['analyze_button']
        self.introduction_button = layout.ids['introduction_button']
        self.file_download_button = layout.ids['file_download_button']
        self.contact_us_button = layout.ids['contact_us_button']

        return layout

    def open_filechooser(self, instance):
        open_filechooser(self, instance)

    def analyze_and_display_results(self, instance):
        self.analysis_result = analyze_and_display_results(self, instance)
        self.analysis_data = self.analysis_result  

    def show_about_page(self, instance):
        show_about_page(self, instance)

    def show_file_download(self, _):
        if not self.analysis_data:
            self.show_error_popup("Please identify first.")
            return
        if not self.analysis_data:
            self.analysis_data = {"file_size": 0, "num_rows": 0, "modification_time": "N/A"}
        show_file_download_popup(self.analysis_data, self.analysis_result)

    def show_contact_us(self, instance):
        show_contact_us_popup()

    def show_error_popup(self, message):
        content = Label(text=message, text_size=(400, None), size_hint=(None, None), halign='center', valign='middle')
        content.bind(texture_size=content.setter('size'))
        popup = Popup(title='Error',
                      content=content,
                      size_hint=(None, None), size=(400, 200),
                      auto_dismiss=True)
        popup.open()

if __name__ == '__main__':
    VideoAnalyzerApp().run()
