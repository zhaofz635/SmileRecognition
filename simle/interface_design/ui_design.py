from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput
from simle.interface_operation.send_email import send_email

class StyledButton(Button):
    pass

def create_main_layout(app_instance):
    layout = FloatLayout()

    main_box = BoxLayout(orientation='vertical', size_hint=(1, 0.9), pos_hint={'top': 1})

    with main_box.canvas.before:
        Color(0.23, 0.35, 0.60, 1)
        main_box.background = Rectangle(size=main_box.size, pos=main_box.pos)

    main_box.bind(size=lambda instance, value: setattr(main_box.background, 'size', value))
    main_box.bind(pos=lambda instance, value: setattr(main_box.background, 'pos', value))

    menu_bar = BoxLayout(size_hint_y=None, height=50)

    introduction_button = Button(text='Introduction',background_color=(0.23, 0.35, 0.60, 1),font_size='23sp')
    introduction_button.bind(on_press=app_instance.show_about_page)
    menu_bar.add_widget(introduction_button)


    contact_us_button = Button(text='ContactUs', background_color=(0.23, 0.35, 0.60, 1), font_size='23sp')
    contact_us_button.bind(on_release=app_instance.show_contact_us)
    menu_bar.add_widget(contact_us_button)

    # 创建 FileDownload 按钮
    file_download_button = Button(text='File Download',  background_color=(0.23, 0.35, 0.60, 1), font_size='23sp')
    file_download_button.bind(on_press=app_instance.show_file_download)
    menu_bar.add_widget(file_download_button)

    main_box.add_widget(menu_bar)


    main_box.add_widget(Label(text='Identify your laughter and smiley faces from videos', size_hint=(1, 0.1), color=(1, 1, 1, 1), font_size='20sp'))

    button_layout = BoxLayout(size_hint=(1, 0.1))
    recognize_button = StyledButton(text='Identifying', size_hint=(0.5, 1), background_color=(0.23, 0.35, 0.60, 1), font_size='40sp')
    recognize_button.bind(on_press=app_instance.open_filechooser)

    analyze_button = StyledButton(text='Analysis', size_hint=(0.5, 1), background_color=(0.23, 0.35, 0.60, 1), font_size='40sp')
    analyze_button.bind(on_press=app_instance.analyze_and_display_results)
    button_layout.add_widget(recognize_button)
    button_layout.add_widget(analyze_button)
    main_box.add_widget(button_layout)

    explanation_layout = BoxLayout(size_hint=(1, 0.1))
    main_box.add_widget(explanation_layout)

    copyright_text = ('Copyright@zhaofz')
    copyright_label = Label(
        text=copyright_text,
        size_hint=(1, 0.1),
        halign='center',
        valign='middle',
        color=(1, 1, 1, 1),
        pos_hint={'x': 0, 'y': 0}
    )
    layout.add_widget(main_box)
    layout.add_widget(copyright_label)

    layout.ids = {
        'recognize_button': recognize_button,
        'analyze_button': analyze_button,
        'introduction_button': introduction_button,
        'file_download_button': file_download_button,  # 添加FileDownload按钮到ids字典中
        'contact_us_button': contact_us_button  # 添加contact_us_button按钮到ids字典中
    }

    return layout

def show_contact_us_popup():
    layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

    contact_info_label = Label(
        text="Team Contact Information:\n\n"
             "Website: https://zhaofz635.github.io/github.io/\n"
             "Email: zhaofz635@gmail.com",
        size_hint=(1, None),
        height=200,
        font_size='16sp'
    )
    layout.add_widget(contact_info_label)

    email_label = Label(
        text="Send Email:",
        size_hint=(1, None),
        height=40,
        font_size='20sp',
        bold=True
    )
    layout.add_widget(email_label)

    email_input = TextInput(
        hint_text="Enter your message here",
        size_hint=(1, 1),
        multiline=True
    )
    layout.add_widget(email_input)

    def send_email_callback(instance):
        message = email_input.text.strip()
        if message:
            send_email("zhaofz635@gmail.com", message)
            popup.dismiss()

    send_button = Button(
        text="Send",
        size_hint=(1, None),
        height=40,
        background_color=(0.1, 0.5, 0.8, 1),
        color=(1, 1, 1, 1)
    )
    send_button.bind(on_release=send_email_callback)
    layout.add_widget(send_button)

    popup = Popup(
        title='Contact Us',
        content=layout,
        size_hint=(None, None),
        size=(Window.width * 0.8, Window.height * 0.8),
        auto_dismiss=True
    )
    popup.open()

