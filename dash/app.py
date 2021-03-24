import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import json
from skimage import io as skio
import io
import base64
import matplotlib.pyplot as plt
import PIL.Image
import pickle
from time import time, sleep
from joblib import Memory
import pims
import pathlib
import plot_common 
import skimage.exposure
import ml_tasks.tasks as ml_tasks 
from dash import callback_context
from kubernetes import client, config
import yaml
import jl # kubernetes job library
from numpy import random

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks = True)

server = app.server
app.title = "Flexible MLexchange"


DEFAULT_IMAGE_PATH = pathlib.Path('../data/input/sample_data.tiff')
seg_path = '/data/out/seg-out.png'
SEG_IMAGE_PATH = pathlib.Path(seg_path)
try:
    SEG_IMAGE_PATH.unlink() #make sure we have a clean directory
except FileNotFoundError:
    pass
    # file doesn't exist, we good, move on
DEFAULT_IMAGE = PIL.Image.open(DEFAULT_IMAGE_PATH)
def make_default_figure(
    #images=[plot_common.img_array_to_pil_image(image_stack[19])],
    images = [DEFAULT_IMAGE],
    shapes=[],
    process_func = lambda x: plot_common.img_array_to_pil_image(
                skimage.exposure.equalize_hist(
                plot_common.pil_image_to_ndarray(x)) )
):
    fig = plot_common.dummy_fig()
    proc_images = [process_func(im) for im in images]
    print(proc_images)
    plot_common.add_layout_images_to_fig(fig, proc_images)
    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
        }
    )
    return fig

def store_shapes_seg_pair(d, key, seg, remove_old=True):
    """
    Stores shapes and segmentation pair in dict d
    seg is a PIL.Image object
    if remove_old True, deletes all the old keys and values.
    """
    bytes_to_encode = io.BytesIO()
    seg.save(bytes_to_encode, format="png")
    bytes_to_encode.seek(0)
    data = base64.b64encode(bytes_to_encode.read()).decode()
    if remove_old:
        return {key: data}
    d[key] = data
    return d




button_github = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/aasgreen/mlexflex",
    id="gh-link",
    style={"text-transform": "none"},
)

# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("MlexFlex"),
                                    html.P("With Three Image Segmentation Backends"),
                                ],
                                id="app-title",
                            )
                        ],
                        md=True,
                        align="center",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(button_github),
                                    ],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    sticky="top",
)

# Image Segmentation
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader("I Can't Believe It's Not Meat!"),
            dbc.CardBody(
                [
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="transparent-loader-wrapper",
                        children=[
                            dcc.Loading(
                                id="segmentations-loading",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="graph",
                                        figure=make_default_figure(),
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawrect",
                                                "drawopenpath",
                                                "eraseshape",
                                            ]
                                        },
                                    ),
                                ],
                            )
                        ],
                    ),
                ]
            ),
            dbc.CardFooter(
                [
                    html.Div(
                        children=[
                                dbc.Row(
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Scipy Segmenter",
                                                id='scipy',
                                                outline=True,
                                                ),
                                            dbc.Button(
                                                'Random Forest Segmenter',
                                                id='random-forest',
                                                outline=True,
                                                ),
                                            dbc.Button(
                                                'MSDNetwork',
                                                id='msdnetwork',
                                                outline=True,
                                                ),
                                        ],
                                    ),
                                ),
                                dbc.Row(
                                    html.H4(
                                        id='return',
                                         children = '',
                                        ), justify='center',
                                    ),
                                ]
                            ),
                        ],
                    ),
                ]
            ),
        ]
# Image Segmentation
results= [
    dbc.Card(
        id="results-card",
        children=[
            dbc.CardHeader("Look at this incredible segmentation!"),
            dbc.CardBody(
                [
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="results-transparent-loader-wrapper",
                        children=[
                            dcc.Loading(
                                id="results-loading",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="results-graph",
                                        figure=make_default_figure(),
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawrect",
                                                "drawopenpath",
                                                "eraseshape",
                                            ]
                                        },
                                    ),
                                ],
                            )
                        ],
                    ),
                ]
            ),
            dbc.CardFooter(
                [
                    html.Div(
                        children=[
                                dbc.Row(
                                            dbc.Button(
                                                "Download Image",
                                                id='dl-image',
                                                outline=True,
                                                ),
                                ),
                                ]
                            ),
                        ],
                    ),
                ]
            ),
        ]


meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created masks
            # data is a list of dicts describing shapes
            dcc.Store(id="masks", data={"shapes": []}),
            dcc.Store(id="classifier-store", data={}),
            dcc.Store(id="classifier-store-temp", data={}),
            dcc.Store(id="classified-image-store", data=""),
            dcc.Store(id="features_hash", data=""),
            dcc.Store(id='current-image-num', data=0),
        ],
    ),
    html.Div(id="download-dummy"),
    html.Div(id="download-image-dummy"),
]

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                html.Div(
                    id="app-content",
                    children= [dbc.Row([dbc.Col(segmentation, width='auto', align='center'),dbc.Col(results, width='auto')], justify='center'), dbc.Row(dbc.Col(meta)),]
                ),
            ],
            fluid=True,
        ),
    ]
)

row = html.Div(
    [
        dbc.Row(dbc.Col(html.Div("A single column"))),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ]
        ),
    ]
)

@app.callback(
        [
            Output("return", "children"),
            Output("results-graph", "figure"),
            ],
        [ Input('scipy', 'n_clicks'),
          Input('random-forest', 'n_clicks'),
          Input('msdnetwork', 'n_clicks'),
          ]
        )
def launch_ml_task(*args):
    trigger = callback_context.triggered[0]
    print('Launching {}'.format(trigger['prop_id']))
    trig_msg = json.dumps(trigger, indent=2)
    print(trig_msg)
#    ml_tasks.hello()
#    res = ml_tasks.hello.delay()
#    ll = res.get(timeout = 1)
#    print('successful: {}'.format(ll))
    succeeded = False
    if trigger['prop_id'] == "scipy.n_clicks":
        print('launching scipy container...')
        job_name = 'segmenting-j{}'.format(random.rand())
        try:
            api, job = jl.main(job_name)
            config.load_incluster_config()

            print('job name: {}'.format(job_name))
            # type(api) is BatchV1Api has method list_job_for_all_namespaces 
            while succeeded == False:
                ret = api.read_namespaced_job(name=job_name, namespace = 'default')
                print(ret.status.succeeded)
                if ret.status.succeeded == True:
                    succeeded = True
                else:
                    sleep(10)
        except Exception as e:
            print('no no{}'.format(e))
            # first pass, I think it will just continue on and not wait for a succeed. We need to use a while loop.
            # if the job returned, then we should be able to access the segmented image
        try:
            seg_img = PIL.Image.open(seg_path)

            seg_figure = make_default_figure([seg_img])
        except Exception as e:
            print('not loaded: {}'.format(e))
        print('succeeded!')
        if succeeded == True:

            print('yay')
            return ['Job Finished',
                    seg_figure]
    return ['', '']


#app.layout = dbc.Container(
#        row,
#        fluid = True
#        )
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
