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
from time import time
from joblib import Memory
import pims
import pathlib
import plot_common 
import skimage.exposure
import ml_tasks.ml_tasks as ml_tasks 

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.title = "Flexible MLexchange"


DEFAULT_IMAGE_PATH = pathlib.Path('../data/input/sample_data.tiff')
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
                                    html.h4(
                                        id='return'
                                        children = ''
                                        ), justify='center',
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
                    children= [dbc.Row(dbc.Col(segmentation, width='auto', align='center'), justify='center'), dbc.Row(dbc.Col(meta)),]
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
        [Ou
#app.layout = dbc.Container(
#        row,
#        fluid = True
#        )
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
