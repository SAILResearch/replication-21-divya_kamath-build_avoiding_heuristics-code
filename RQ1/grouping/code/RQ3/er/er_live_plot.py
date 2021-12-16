from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, RadioButtonGroup
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
import pickle

from utils import er_project_list

project_name = 'bb'

project_name_dict = {
    0: 'bb',
    1: 'ebb',
    2: 'racoam',
}


def modify_doc(doc):
    with open('er_batch_size.pickle', 'rb') as handle:
        project_dict = pickle.load(handle)

    data = project_dict['bb'][0.1]
    source = ColumnDataSource(data)

    plot = figure(x_axis_type='linear', y_range=(-200, 200), y_axis_label='Improvement (%)', x_axis_label='Batch Size')
    plot.line('x', 'y', source=source)

    def callback(attr, old, new):
        global project_name

        if attr == 'active':
            data = project_dict[project_name_dict[new]][0.1]
            project_name = project_name_dict[new]

        else:
            new = float('{:2f}'.format(new))
            data = project_dict[project_name][new]

        source.data = ColumnDataSource(data=data).data

    slider = Slider(start=0.1, end=1, value=0.1, step=0.01, title="Risk Threshold")
    slider.on_change('value', callback)

    radio_button_group = RadioButtonGroup(
        labels=er_project_list, active=0)

    radio_button_group.on_change('active', callback)

    doc.add_root(column(radio_button_group, slider, plot))

    doc.theme = Theme(filename="../theme.yaml")


# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({
    '/': modify_doc,
}, num_procs=1,
    allow_websocket_origin=['127.0.0.1:8000', 'localhost:5006', '127.0.0.1'])

server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
