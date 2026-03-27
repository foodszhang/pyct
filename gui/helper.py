import dearpygui.dearpygui as dpg 
class LayoutHelper:
    def __init__(self):
        self.table_id = dpg.add_table(header_row=False, policy=dpg.mvTable_SizingStretchProp)
        self.stage_id = dpg.add_stage()
        dpg.push_container_stack(self.stage_id)
        
    def add_widget(self, uuid, percentage):
        dpg.add_table_column(init_width_or_weight=percentage/100.0, parent=self.table_id)
        try:
            dpg.set_item_width(uuid, -1)
        except:
            pass

    def submit(self):
        dpg.pop_container_stack() # pop stage
        with dpg.table_row(parent=self.table_id):
            dpg.unstage(self.stage_id)

def on_selection(sender, unused, user_data):

    if user_data[1]:
        print("User selected 'Ok'")
    else:
        print("User selected 'Cancel'")

    # delete window
    dpg.delete_item(user_data[0])

def show_info(title, message, selection_callback=on_selection):

    # guarantee these commands happen in the same frame
    with dpg.mutex():

        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        with dpg.window(label=title, modal=True, no_close=True) as modal_id:
            dpg.add_text(message)
            dpg.add_button(label="Ok", width=75, user_data=(modal_id, True), callback=selection_callback)
            dpg.add_same_line()
            dpg.add_button(label="Cancel", width=75, user_data=(modal_id, False), callback=selection_callback)

    # guarantee these commands happen in another frame
    dpg.split_frame()
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    assert width and height
    dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])
