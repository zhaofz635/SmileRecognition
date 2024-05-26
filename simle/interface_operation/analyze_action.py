import pandas as pd
import os
from datetime import datetime
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout

class AnalysisWindow(BoxLayout):
    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        scroll_view = ScrollView(size_hint=(1, 1))
        self.add_widget(scroll_view)

        grid_layout = GridLayout(cols=len(df.columns), size_hint_y=None)
        grid_layout.bind(minimum_height=grid_layout.setter('height'))
        scroll_view.add_widget(grid_layout)

        for column_name in df.columns:
            label = Label(text=column_name, size_hint_y=None, height=50)
            grid_layout.add_widget(label)

        for index, row in df.iterrows():
            for value in row:
                label = Label(text=str(value), size_hint_y=None, height=50)
                grid_layout.add_widget(label)

def analyze_and_display_results(app_instance, instance):
    if not hasattr(app_instance, 'recognition_results'):
        print("No recognition results available for analysis")
        return None

    try:
        laughter_data = pd.read_csv("laughter_data.csv")
    except FileNotFoundError:
        print("Analysis report file not found")
        return None

    analysis_window = AnalysisWindow(laughter_data)
    popup = Popup(title='Analysis Results', content=analysis_window, size_hint=(0.9, 0.9))
    popup.open()

    # Get file details
    file_info = os.stat("laughter_data.csv")
    file_size = file_info.st_size
    num_rows = len(laughter_data)
    modification_time = datetime.fromtimestamp(file_info.st_mtime)

    analysis_details = {
        "file_size": file_size,
        "num_rows": num_rows,
        "modification_time": modification_time
    }

    return analysis_details
