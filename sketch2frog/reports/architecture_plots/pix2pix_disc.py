
# open git bash in top level directory
# cd networks_go_here
# bash ../tikzmake.sh pix2pix_disc

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *
from functools import partial

# these 4 all all the same ezxept for the colour... this could be simplified
def _layer_wrapper(name, fill, s_filer=" ", n_filer=" ", offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        zlabel="""+ str(s_filer) +""",
        fill="""+ fill +r""",
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

to_BN = partial(_layer_wrapper, fill=r"""\BatchNColor""")
to_Relu = partial(_layer_wrapper, fill=r"""\ReluColor""")
to_LeakyRelu = partial(_layer_wrapper, fill=r"""\LeakyReluColor""")
to_Dropout = partial(_layer_wrapper, fill=r"""\DropoutColor""")

def to_Concat( name, offset="(0,0,0)", to="(0,0,0)", radius=2.5):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Ball={
        name=""" + name + """,
        fill=\ConcatColor,
        radius="""+ str(radius) +""",
        logo=$||$
        }
    };
"""

def to_Output( pathfile, x_position=35.5, to='(3,0,0)', width=8, height=8, name="temp" ):
    return r"""
\node[canvas is zy plane at x=""" + str(x_position) + """] (""" + name + """) at """+ to +"""
    {\includegraphics[width="""+ str(width)+"cm"+""",height="""+ str(height)+"cm"+"""]{"""+ pathfile +"""}};
"""

arch = [ 
    to_head('..'), 
    to_cor(), # define colours # tanh is purple
    r"""\def\ConcatColor{rgb:blue,5;red,2.5;white,5}""",
    r"""\def\BatchNColor{rgb:red,1;black,0.3}""", # red
    r"""\def\ReluColor{rgb:yellow,5;red,5;white,5}""", # darker conv
    r"""\def\LeakyReluColor{rgb:yellow,5;red,5;white,5}""", # same as relu
    r"""\def\DropoutColor{rgb:blue,5;green,15}""", # green
    r"""\def\ConvTransposedColor{rgb:blue,2;green,1;black,0.3}""", # light blue
    to_begin(),
    
    #######################################################################################
    ### Draw Input
    #######################################################################################
    # to_MultiInput('../networks_go_here/pix2pix Generator/frog-728.png', x_position=0, y_position=0, width=8, height=8, name='input1'),
    # to_MultiInput('../networks_go_here/pix2pix Generator/frog-728-pred.png', to="(2,2,0)", name="input2" ),
    r"""
\node[canvas is zy plane at x=0, shift={(0,-5,0)}] (input1) at (0,0,0)
    {\includegraphics[width=8cm,height=8cm]{../networks_go_here/pix2pix/frog-728.png}};
\node[canvas is zy plane at x=0, shift={(0,10,0)}] (input2) at (input1)
    {\includegraphics[width=8cm,height=8cm]{../networks_go_here/pix2pix/frog-728-pred.png}};
""",
    
    to_Concat(name="cat_input", offset="(3,5,0)", to="(input1)", radius=2.5),
    
    #######################################################################################
    ### Draw Encoder
    #######################################################################################
   
    # conv1
    to_Conv(name="c1", s_filer=" ", n_filer=64, offset="(3,0,0)", to="(cat_input-east)", width=2, height=49, depth=49, caption=" " ),
    to_LeakyRelu(name="lr1", s_filer="256x256", offset="(0,0,0)", to="(c1-east)", width=1, height=49, depth=49, opacity=0.6, caption=" "),
    # conv2
    to_Conv(name="c2", s_filer=" ", n_filer=128, offset="(4,0,0)", to="(lr1-east)", width=4, height=36, depth=36, caption=" " ),
    to_BN(name="bn2", offset="(0,0,0)", to="(c2-east)", width=1, height=36, depth=36, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr2", s_filer="128x128", offset="(0,0,0)", to="(bn2-east)", width=1, height=36, depth=36, opacity=0.6, caption=" "),
    # conv3
    to_Conv(name="c3", s_filer=" ", n_filer=256, offset="(3,0,0)", to="(lr2-east)", width=8, height=25, depth=25, caption=" " ),
    to_BN(name="bn3", offset="(0,0,0)", to="(c3-east)", width=1, height=25, depth=25, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr3", s_filer="64x64", offset="(0,0,0)", to="(bn3-east)", width=1, height=25, depth=25, opacity=0.6, caption=" "),
    # conv4
    to_Conv(name="c4", s_filer=" ", n_filer=512, offset="(2,0,0)", to="(lr3-east)", width=16, height=16, depth=16, caption=" " ),
    to_BN(name="bn4", offset="(0,0,0)", to="(c4-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr4", s_filer="32x32", offset="(0,0,0)", to="(bn4-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    # conv5 no strides
    to_Conv(name="c5", s_filer=" ", n_filer=512, offset="(2,0,0)", to="(lr4-east)", width=16, height=16, depth=16, caption="No strides" ),
    to_BN(name="bn5", offset="(0,0,0)", to="(c5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr5", s_filer="16x16", offset="(0,0,0)", to="(bn5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),


    #######################################################################################
    ### Classifier
    #######################################################################################
    # patch out
    to_Conv(name="c6", s_filer=" ", n_filer=1, offset="(2,0,0)", to="(lr5-east)", width=1, height=16, depth=16, caption="" ),
    
    #######################################################################################
    ### Draw Output
    #######################################################################################
    to_Output( "../networks_go_here/pix2pix/patch.png", x_position=30, to='(3,0,0)', width=3.5, height=3.5),
    
    #######################################################################################
    ### Draw Legend
    #######################################################################################
    to_Conv(name="c_legend", s_filer=" ", n_filer=" ", offset="(0,-5,0)", to="(c4-south)", width=10, height=10, depth=10, caption="Conv" ),
    to_BN(name="bn_legend", offset="(1.25,0,0)", to="(c_legend-east)", width=1, height=10, depth=10, opacity=0.6, caption="Batch Norm"),
    to_LeakyRelu(name="lr_legend", offset="(1.25,0,0)", to="(bn_legend-east)", width=1, height=10, depth=10, opacity=0.6, caption="Leaky ReLU"),

    #######################################################################################
    ### Draw connections
    #######################################################################################
    # defines the shapes crpX-mid

    r"""
\path (input1) -- (cat_input-west|-input1) coordinate[pos=0.5] (input1cat-mid) ;
\path (input2) -- (cat_input-west|-input2) coordinate[pos=0.5] (input2cat-mid) ;
""",
    # arrows pointing down the U. Overlaps the arrow from left to concat
    # consists of 3 lines with 3 pointers
    r"""
\draw[connection](input1)--node{}(input1cat-mid)--node{\midarrow}(cat_input-west-|input1cat-mid)--node{}(cat_input-west);
\draw[connection](input2)--node{}(input2cat-mid)--node{\midarrow}(cat_input-west-|input2cat-mid)--node{}(cat_input-west);
""",
    # arrows on RHS between convconvrelu and upconv
    # only keeping arrow to softmax
#     r"""
# \draw [connection]  (c6-east) -- node {\midarrow} (out-west);
# """,
    # arrow between LHS and concat symbol
    r"""
\draw [copyconnection]  (cat_input-east)  -- node {\copymidarrow} (c1-west);
\draw [copyconnection]  (lr1-east)  -- node {\copymidarrow} (c2-west);
\draw [copyconnection]  (lr2-east)  -- node {\copymidarrow} (c3-west);
\draw [copyconnection]  (lr3-east)  -- node {\copymidarrow} (c4-west);
\draw [copyconnection]  (lr4-east)  -- node {\copymidarrow} (c5-west);
\draw [copyconnection]  (lr5-east)  -- node {\copymidarrow} (c6-west);
""",
    
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()