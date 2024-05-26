from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle, Line

class BorderedBox(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding = [20, 20, 20, 20]  
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1) 
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)
        with self.canvas.after:
            Color(1, 1, 1, 1) 
            self.line = Line(rectangle=[self.x, self.y, self.width, self.height], width=1.5)
            self.bind(size=self._update_line, pos=self._update_line)

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def _update_line(self, instance, value):
        self.line.rectangle = [self.x, self.y, self.width, self.height]

def show_about_page(instance, *args):
    # Main layout
    main_layout = BoxLayout(orientation='vertical', padding=20, spacing=20, size_hint_y=None)
    main_layout.bind(minimum_height=main_layout.setter('height'))

    def create_bordered_label(text, height):
        box = BorderedBox(orientation='vertical', size_hint=(1, None), height=height)
        label = Label(
            text=text,
            size_hint=(1, None),
            height=height - 40,  
            valign='top',
            halign='left',
            text_size=(Window.width * 0.8, None),
            color=(0.9, 0.9, 0.9, 1),
            markup=True
        )
        box.add_widget(label)
        return box, label

    # Software function introduction
    functionality_box, software_functionality = create_bordered_label(
        '[b]Software Functionality:[/b]\nThis software provides a platform for identifying laughter and smiling faces from videos using deep learning techniques.',
        200
    )

   # Team information
    team_box = BorderedBox(orientation='horizontal', size_hint=(1, None), height=220, spacing=20)
    team_member_image = Image(source='/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/image/WechatIMG63.jpg', size_hint=(None, None), size=(200, 200))
    team_member_info = Label(
        text='[b]Software Development Team:[/b]\nFuzheng ZHAO\nLead Developer\n',
        valign='top',
        halign='left',
        text_size=(Window.width * 0.6, None),
        color=(0.9, 0.9, 0.9, 1),
        markup=True
    )
    team_box.add_widget(team_member_image)
    team_box.add_widget(team_member_info)

    # Core Technology
    technology_box, core_technology = create_bordered_label(
        '[b]Core Technology:[/b]\n- Deep fusion neural networks\n- Feature extraction\n- Real-time analysis',
        150
    )

    # Add components to the main layout
    main_layout.add_widget(functionality_box)
    main_layout.add_widget(team_box)
    main_layout.add_widget(technology_box)

    # Use scroll view to wrap the main layout
    scroll_view = ScrollView(size_hint=(1, 1))
    scroll_view.add_widget(main_layout)

    #Create and open popup window
    about_popup = Popup(title='About', content=scroll_view, size_hint=(0.95, 0.95), auto_dismiss=True)
    about_popup.open()

    # Bind window size change event
    Window.bind(on_resize=lambda instance, width, height: adjust_text_size(software_functionality, core_technology))

def adjust_text_size(*labels):
    for label in labels:
        label.text_size = (Window.width * 0.8, None)

class TestApp(App):
    def build(self):
        root = BoxLayout(orientation='vertical')

        with root.canvas.before:
            Color(0.1, 0.1, 0.1, 1)  
            self.rect = Rectangle(size=root.size, pos=root.pos)
            root.bind(size=self._update_rect, pos=root.pos)

        btn = Button(text="Show About Page", size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        btn.bind(on_release=show_about_page)
        root.add_widget(btn)
        return root

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

if __name__ == "__main__":
    TestApp().run()
