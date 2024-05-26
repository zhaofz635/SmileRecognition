from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import os
# 使用相对路径导入 recognize_video 模块
from ..interface_operation import recognize_video_old

def open_filechooser(app_instance, instance):
    user_home = os.path.expanduser('~')
    content = BoxLayout(orientation='vertical', spacing=10)
    app_instance.file_chooser = FileChooserIconView(path=user_home, filters=['*.mp4', '*.avi'])
    content.add_widget(app_instance.file_chooser)

    button_layout = BoxLayout(size_hint_y=None, height=50)
    select_button = app_instance.root.ids.recognize_button.__class__(text="Select", size_hint=(0.5, 1), background_color=(0.23, 0.35, 0.60, 1))
    select_button.bind(on_press=lambda x: select_file(app_instance))
    cancel_button = app_instance.root.ids.recognize_button.__class__(text="Cancel", size_hint=(0.5, 1), background_color=(0.23, 0.35, 0.60, 1))
    cancel_button.bind(on_press=lambda x: app_instance.popup.dismiss())
    button_layout.add_widget(select_button)
    button_layout.add_widget(cancel_button)

    content.add_widget(button_layout)

    app_instance.popup = Popup(title='Select a video file', content=content, size_hint=(0.9, 0.9))
    app_instance.popup.open()

def select_file(app_instance):
    selected_files = app_instance.file_chooser.selection
    if not selected_files:
        print("No file selected")
        app_instance.popup.dismiss()
        return

    file_path = selected_files[0]
    print("Selected file:", file_path)

    app_instance.recognition_results = recognize_video.recognize_laughter(file_path)
    print("Recognition completed, result:", app_instance.recognition_results)
    app_instance.popup.dismiss()
