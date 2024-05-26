from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window
import os
import io
import csv

def preprocess_data(data):
    """Ensure all elements in the data are dictionaries and replace None values with 0."""
    if not data:
        return [{'FileName': 0, 'LaughingBehavior': 0, 'BehaviorType': 0, 'StartingTime': 0, 'EndingTime': 0}]

    processed_data = []
    for row in data:
        if isinstance(row, dict):
            processed_row = {k: (0 if v is None else v) for k, v in row.items()}
            processed_data.append(processed_row)
    return processed_data

def convert_to_csv(data):
    """Convert a list of dictionaries to a CSV string."""
    data = preprocess_data(data)  # Ensure data is preprocessed before conversion
    if not data or not isinstance(data[0], dict):
        data = [{'FileName': 0, 'LaughingBehavior': 0, 'BehaviorType': 0, 'StartingTime': 0, 'EndingTime': 0}]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()

def show_file_download_popup(analysis_data, analysis_result):
    layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

    overview_label = Label(
        text="CSV Overview:",
        size_hint=(1, None),
        height=40,
        font_size='20sp',
        bold=True
    )
    layout.add_widget(overview_label)

    details_text = (
        f"File Size: {analysis_data['file_size']} bytes\n"
        f"Number of Rows: {analysis_data['num_rows']}\n"
        f"Last Modified: {analysis_data['modification_time']}\n"
    )
    overview_text = Label(
        text=details_text,
        size_hint=(1, None),
        height=120,
        font_size='16sp'
    )
    layout.add_widget(overview_text)

    save_status_label = Label(
        text="",
        size_hint=(1, None),
        height=40,
        font_size='14sp'
    )
    layout.add_widget(save_status_label)

    selected_path_label = Label(
        text="Save Path: ",
        size_hint=(1, None),
        height=40,
        font_size='14sp'
    )
    layout.add_widget(selected_path_label)

    def open_filechooser(instance):
        filechooser_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        filechooser = FileChooserListView(size_hint=(1, 1), path=os.path.expanduser('~'), dirselect=True)

        def select_path(instance):
            selection = filechooser.selection
            if selection:
                selected_path = selection[0]
                selected_path_label.text = f"Save Path: {selected_path}"
                filechooser_popup.dismiss()
                open_filename_input_popup(selected_path)

        filechooser_layout.add_widget(filechooser)

        select_button = Button(
            text="Select",
            size_hint=(1, None),
            height=40,
            background_color=(0.1, 0.5, 0.8, 1),
            color=(1, 1, 1, 1)
        )
        select_button.bind(on_release=select_path)
        filechooser_layout.add_widget(select_button)

        filechooser_popup = Popup(
            title="Select Directory",
            content=filechooser_layout,
            size_hint=(0.9, 0.9)
        )
        filechooser_popup.open()

    def open_filename_input_popup(selected_path):
        filename_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        file_name_input = TextInput(
            hint_text="Enter file name (including .csv extension)",
            size_hint=(1, None),
            height=40,
            multiline=False
        )
        filename_layout.add_widget(file_name_input)

        def save_file(instance):
            file_name = file_name_input.text.strip()
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            file_path = os.path.join(selected_path, file_name)

            try:
                # Convert analysis_result to CSV string
                csv_data = convert_to_csv(analysis_result)

                # Save the CSV string to a file
                with open(file_path, 'w', newline='') as f:
                    f.write(csv_data)
                save_status_label.text = f"File saved at: {file_path}"
                save_status_label.color = (0, 1, 0, 1)  # Green color for success status
            except Exception as e:
                save_status_label.text = f"Failed to save file: {str(e)}"
                save_status_label.color = (1, 0, 0, 1)  # Red color for error status

            filename_popup.dismiss()  # Close the popup after file save

        save_button = Button(
            text="Save",
            size_hint=(1, None),
            height=40,
            background_color=(0.1, 0.5, 0.8, 1),
            color=(1, 1, 1, 1)
        )
        save_button.bind(on_release=save_file)
        filename_layout.add_widget(save_button)

        filename_popup = Popup(
            title="Enter File Name",
            content=filename_layout,
            size_hint=(0.7, 0.3)
        )
        filename_popup.open()

    save_button = Button(
        text="Save File",
        size_hint=(1, None),
        height=40,
        background_color=(0.1, 0.5, 0.8, 1),
        color=(1, 1, 1, 1)
    )
    save_button.bind(on_release=open_filechooser)
    layout.add_widget(save_button)

    popup = Popup(
        title='File Download',
        content=layout,
        size_hint=(None, None),
        size=(Window.width * 0.8, Window.height * 0.8),
        auto_dismiss=True,
        pos_hint={'center_x': 0.5, 'center_y': 0.5}  # Set popup position to center
    )
    popup.open()

# Test data for demonstration
analysis_data = {
    'file_size': 108,
    'num_rows': 1,
    'modification_time': '2024-05-26 01:18:49.421730'
}

# Test result for demonstration
analysis_result = [
    {'FileName': None, 'LaughingBehavior': None, 'BehaviorType': 'Behavior detected', 'StartingTime': None, 'EndingTime': None}
]

show_file_download_popup(analysis_data, analysis_result)
