# open git bash in top level directory
# cd networks_go_here
# bash ../tikzmake.sh pix2pix_gen

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *
from functools import partial

# these 4 all all the same ezxept for the colour... this could be simplified
def _layer_wrapper(name, fill, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
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

def to_ConvTransposed( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvTransposedColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# delete if working
if True:
    # # Bat``chNorm
    # def to_BN(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    #     return r"""
    # \pic[shift={ """+ offset +""" }] at """+ to +""" 
    #     {Box={
    #         name="""+name+""",
    #         caption="""+ caption +r""",
    #         fill=\BatchNColor,
    #         opacity="""+ str(opacity) +""",
    #         height="""+ str(height) +""",
    #         width="""+ str(width) +""",
    #         depth="""+ str(depth) +"""
    #         }
    #     };
    # """

    # def to_Relu(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    #     return r"""
    # \pic[shift={ """+ offset +""" }] at """+ to +""" 
    #     {Box={
    #         name="""+name+""",
    #         caption="""+ caption +r""",
    #         fill=\ReluColor,
    #         opacity="""+ str(opacity) +""",
    #         height="""+ str(height) +""",
    #         width="""+ str(width) +""",
    #         depth="""+ str(depth) +"""
    #         }
    #     };
    # """

    # def to_LeakyRelu(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    #     return r"""
    # \pic[shift={ """+ offset +""" }] at """+ to +""" 
    #     {Box={
    #         name="""+name+""",
    #         caption="""+ caption +r""",
    #         fill=\LeakyReluColor,
    #         opacity="""+ str(opacity) +""",
    #         height="""+ str(height) +""",
    #         width="""+ str(width) +""",
    #         depth="""+ str(depth) +"""
    #         }
    #     };
    # """

    # def to_Dropout(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    #     return r"""
    # \pic[shift={ """+ offset +""" }] at """+ to +""" 
    #     {Box={
    #         name="""+name+""",
    #         caption="""+ caption +r""",
    #         fill=\DropoutColor,
    #         opacity="""+ str(opacity) +""",
    #         height="""+ str(height) +""",
    #         width="""+ str(width) +""",
    #         depth="""+ str(depth) +"""
    #         }
    #     };
    # """``
    pass


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

def to_ConvRelu( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name +""",
        caption="""+ caption +""",
        xlabel={{ """+ str(n_filer) +""", "dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width=""" + str(width) + """,
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Output( pathfile, x_position=35.5, to='(3,0,0)', width=8, height=8, name="temp" ):
    return r"""
\node[canvas is zy plane at x=""" + str(x_position) + """] (""" + name + """) at """+ to +"""
    {\includegraphics[width="""+ str(width)+"cm"+""",height="""+ str(height)+"cm"+"""]{"""+ pathfile +"""}};
"""

    
#     #######################################################################################
#     ### Draw Output
#     #######################################################################################
#     to_Output( "../networks_go_here/UNet256/frog-728-pred.png", x_position=35, to='(3,0,0)', width=8, height=8, name="temp" ),

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
    to_input( '../networks_go_here/pix2pix Generator/frog-728.png', width=12, height=12 ),
    
    #######################################################################################
    ### Draw Encoder
    #######################################################################################
   
    # conv1
    to_Conv(name="c1", s_filer=" ", n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=2, height=45, depth=45, caption=" " ),
    to_LeakyRelu(name="lr1", offset="(0,0,0)", to="(c1-east)", width=1, height=45, depth=45, opacity=0.6, caption=" "),
    # conv2
    to_Conv(name="c2", s_filer=" ", n_filer=128, offset="(1.2,-10,0)", to="(lr1-east)", width=3.5, height=40, depth=40, caption=" " ),
    to_BN(name="bn2", offset="(0,0,0)", to="(c2-east)", width=1, height=40, depth=40, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr2", offset="(0,0,0)", to="(bn2-east)", width=1, height=40, depth=40, opacity=0.6, caption=" "),
    # conv3   
    to_Conv(name="c3", s_filer=" ", n_filer=256, offset="(1.2,-8.5,0)", to="(lr2-east)", width=4.5, height=32, depth=32, caption=" " ),
    to_BN(name="bn3", offset="(0,0,0)", to="(c3-east)", width=1, height=32, depth=32, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr3", offset="(0,0,0)", to="(bn3-east)", width=1, height=32, depth=32, opacity=0.6, caption=" "),
    # conv4
    to_Conv(name="c4", s_filer=" ", n_filer=512, offset="(1.2,-6.5,0)", to="(lr3-east)", width=6, height=25, depth=25, caption=" " ),
    to_BN(name="bn4", offset="(0,0,0)", to="(c4-east)", width=1, height=25, depth=25, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr4", offset="(0,0,0)", to="(bn4-east)", width=1, height=25, depth=25, opacity=0.6, caption=" "),
    # conv5
    to_Conv(name="c5", s_filer=" ", n_filer=512, offset="(1.2,-5,0)", to="(lr4-east)", width=7, height=16, depth=16, caption=" " ),
    to_BN(name="bn5", offset="(0,0,0)", to="(c5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr5", offset="(0,0,0)", to="(bn5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    # conv6
    to_Conv(name="c6", s_filer=" ", n_filer=512, offset="(1.2,-3,0)", to="(lr5-east)", width=10, height=9, depth=9, caption=" " ),
    to_BN(name="bn6", offset="(0,0,0)", to="(c6-east)", width=1, height=9, depth=9, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr6", offset="(0,0,0)", to="(bn6-east)", width=1, height=9, depth=9, opacity=0.6, caption=" "),
    # conv7
    to_Conv(name="c7", s_filer=" ", n_filer=512, offset="(1.2,-3,0)", to="(lr6-east)", width=20, height=4, depth=4, caption=" " ),
    to_BN(name="bn7", offset="(0,0,0)", to="(c7-east)", width=1, height=4, depth=4, opacity=0.6, caption=" "),
    to_LeakyRelu(name="lr7", offset="(0,0,0)", to="(bn7-east)", width=1, height=4, depth=4, opacity=0.6, caption=" "),
    
    #######################################################################################
    ### Bottleneck
    #######################################################################################
    to_Conv(name="c8", s_filer=" ", n_filer=1024, offset="(1.2,-3,0)", to="(lr7-east)", width=40, height=2, depth=2, caption=" " ),
    to_Relu(name="r8", offset="(0,0,0)", to="(c8-east)", width=1, height=2, depth=2, opacity=0.6, caption=" "),
    
    #######################################################################################
    ### Draw Decoder
    #######################################################################################
    # transposed conv7
    to_ConvTransposed(name="ct7", s_filer=" ", n_filer=512, offset="(1.4,0,0)", to="(r8-east)", width=20, height=4, depth=4, caption=" " ),
    to_BN(name="bn_7", offset="(0,0,0)", to="(ct7-east)", width=1, height=4, depth=4, opacity=0.6, caption=" "),
    to_Dropout(name="drop_7", offset="(0,0,0)", to="(bn_7-east)", width=1, height=4, depth=4, opacity=0.6, caption=" "),
    to_Concat(name="cat7", offset="(0,3,0)", to="(drop_7-anchor)", radius=2.5),
    to_Relu(name="r_7", offset="(0.7,0,0)", to="(cat7-east)", width=1, height=4, depth=4, opacity=0.6, caption=" "),

    # transposed conv6
    to_ConvTransposed(name="ct6", s_filer=" ", n_filer=512, offset="(0.7,0,0)", to="(r_7-east)", width=10, height=9, depth=9, caption=" " ),
    to_BN(name="bn_6", offset="(0,0,0)", to="(ct6-east)", width=1, height=9, depth=9, opacity=0.6, caption=" "),
    to_Dropout(name="drop_6", offset="(0,0,0)", to="(bn_6-east)", width=1, height=9, depth=9, opacity=0.6, caption=" "),
    to_Concat(name="cat6", offset="(0,3,0)", to="(drop_6-anchor)", radius=2.5),
    to_Relu(name="r_6", offset="(1.4,0,0)", to="(cat6-east)", width=1, height=9, depth=9, opacity=0.6, caption=" "),
    
    # transposed conv5
    to_ConvTransposed(name="ct5", s_filer=" ", n_filer=512, offset="(1.4,0,0)", to="(r_6-east)", width=7, height=16, depth=16, caption=" " ),
    to_BN(name="bn_5", offset="(0,0,0)", to="(ct5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    to_Dropout(name="drop_5", offset="(0,0,0)", to="(bn_5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    to_Concat(name="cat5", offset="(0,3,0)", to="(drop_5-anchor)", radius=2.5),
    to_Relu(name="r_5", offset="(1.4,0,0)", to="(cat5-east)", width=1, height=16, depth=16, opacity=0.6, caption=" "),
    

    # transposed conv4
    to_ConvTransposed(name="ct4", s_filer=" ", n_filer=512, offset="(1.8,0,0)", to="(r_5-east)", width=6, height=25, depth=25, caption=" " ),
    to_BN(name="bn_4", offset="(0,0,0)", to="(ct4-east)", width=1, height=25, depth=25, opacity=0.6, caption=" "),
    to_Concat(name="cat4", offset="(0,5,0)", to="(bn_4-anchor)", radius=2.5),
    to_Relu(name="r_4", offset="(2,0,0)", to="(cat4-east)", width=1, height=25, depth=25, opacity=0.6, caption=" "),
    
    # transposed conv3
    to_ConvTransposed(name="ct3", s_filer=" ", n_filer=256, offset="(2.6,0,0)", to="(r_4-east)", width=4.5, height=32, depth=32, caption=" " ),
    to_BN(name="bn_3", offset="(0,0,0)", to="(ct3-east)", width=1, height=32, depth=32, opacity=0.6, caption=" "),
    to_Concat(name="cat3", offset="(0,6.5,0)", to="(bn_3-anchor)", radius=2.5),
    to_Relu(name="r_3", offset="(2.6,0,0)", to="(cat3-east)", width=1, height=32, depth=32, opacity=0.6, caption=" "),
    
    # transposed conv2
    to_ConvTransposed(name="ct2", s_filer=" ", n_filer=128, offset="(2,0,0)", to="(r_3-east)", width=3.5, height=40, depth=40, caption=" " ),
    to_BN(name="bn_2", offset="(0,0,0)", to="(ct2-east)", width=1, height=40, depth=40, opacity=0.6, caption=" "),
    to_Concat(name="cat2", offset="(0,8.5,0)", to="(bn_2-anchor)", radius=2.5),
    to_Relu(name="r_2", offset="(3,0,0)", to="(cat2-east)", width=1, height=40, depth=40, opacity=0.6, caption=" "),
    
    # transposed conv1
    to_ConvTransposed(name="ct1", s_filer=" ", n_filer=64, offset="(2,0,0)", to="(r_2-east)", width=2, height=45, depth=45, caption=" " ),
    to_BN(name="bn_1", offset="(0,0,0)", to="(ct1-east)", width=1, height=45, depth=45, opacity=0.6, caption=" "),
    to_Concat(name="cat1", offset="(0,10,0)", to="(bn_1-anchor)", radius=2.5),
    to_Relu(name="r_1", offset="(3.4,0,0)", to="(cat1-east)", width=1, height=45, depth=45, opacity=0.6, caption=" "),
    

    #######################################################################################
    ### Classifier
    #######################################################################################
    to_ConvTransposed(name="ct0", s_filer=" ", n_filer=3, offset="(2,0,0)", to="(r_1-east)", width=1, height=60, depth=60, caption=" " ),
    to_SoftMax( name="out", s_filer=" ", offset="(2,0,0)", to="(ct0-east)", width=1, height=60, depth=60, opacity=0.8, caption="tanh" ),
    
    #######################################################################################
    ### Draw Output
    #######################################################################################
    to_Output( "../networks_go_here/pix2pix Generator/frog-728-pred.png", x_position=80, to='(3,0,0)', width=12, height=12),
    
    #######################################################################################
    ### Draw Legend
    #######################################################################################
    to_Conv(name="c_legend", s_filer=" ", n_filer=" ", offset="(0,-10,0)", to="(bn_1-south)", width=10, height=10, depth=10, caption="Conv" ),
    to_ConvTransposed(name="ct_legend", s_filer=" ", n_filer=" ", offset="(2.5,0,0)", to="(c_legend-east)", width=10, height=10, depth=10, caption="Transposed Conv" ),
    to_LeakyRelu(name="lr_legend", offset="(-1.25,-3.5,0)", to="(c_legend-south)", width=1, height=10, depth=10, opacity=0.6, caption="ReLU"),
    to_BN(name="bn_legend", offset="(2.5,0,0)", to="(lr_legend-east)", width=1, height=10, depth=10, opacity=0.6, caption="BatchNorm"),
    to_Dropout(name="drop_legend", offset="(2.5,0,0)", to="(bn_legend-east)", width=1, height=10, depth=10, opacity=0.6, caption="Dropout"),

    #######################################################################################
    ### Draw connections
    #######################################################################################
    # defines the shapes crpX-mid
    # "|-" means arrow is down then across
    r"""
\path (lr1-east) -- (c2-west|-lr1-west) coordinate[pos=0.5] (lr1c2-mid) ;
\path (lr2-east) -- (c3-west|-lr2-west) coordinate[pos=0.5] (lr2c3-mid) ;
\path (lr3-east) -- (c4-west|-lr3-west) coordinate[pos=0.5] (lr3c4-mid) ;
\path (lr4-east) -- (c5-west|-lr4-west) coordinate[pos=0.5] (lr4c5-mid) ;
\path (lr5-east) -- (c6-west|-lr5-west) coordinate[pos=0.5] (lr5c6-mid) ;
\path (lr6-east) -- (c7-west|-lr6-west) coordinate[pos=0.5] (lr6c7-mid) ;
\path (lr7-east) -- (c8-west|-lr7-west) coordinate[pos=0.5] (lr7c8-mid) ;
""",
    # arrows pointing down the U. Overlaps the arrow from left to concat
    # consists of 3 lines with 3 pointers
    r"""
\draw[connection](lr1-east)--node{\midarrow}(lr1c2-mid)--node{\midarrow}(c2-west-|lr1c2-mid)--node{\midarrow}(c2-west);
\draw[connection](lr2-east)--node{\midarrow}(lr2c3-mid)--node{\midarrow}(c3-west-|lr2c3-mid)--node{\midarrow}(c3-west);
\draw[connection](lr3-east)--node{\midarrow}(lr3c4-mid)--node{\midarrow}(c4-west-|lr3c4-mid)--node{\midarrow}(c4-west);
\draw[connection](lr4-east)--node{\midarrow}(lr4c5-mid)--node{\midarrow}(c5-west-|lr4c5-mid)--node{\midarrow}(c5-west);
\draw[connection](lr5-east)--node{\midarrow}(lr5c6-mid)--node{\midarrow}(c6-west-|lr5c6-mid)--node{\midarrow}(c6-west);
\draw[connection](lr6-east)--node{\midarrow}(lr6c7-mid)--node{\midarrow}(c7-west-|lr6c7-mid)--node{\midarrow}(c7-west);
\draw[connection](lr7-east)--node{\midarrow}(lr7c8-mid)--node{\midarrow}(c8-west-|lr7c8-mid)--node{\midarrow}(c8-west);
""",
    # arrows between relu and next transposed conv
    r"""
\draw [connection]  (r8-east) -- node {\midarrow} (ct7-west);
\draw [connection]  (r_7-east) -- node {\midarrow} (ct6-west);
\draw [connection]  (r_6-east) -- node {\midarrow} (ct5-west);
\draw [connection]  (r_5-east) -- node {\midarrow} (ct4-west);
\draw [connection]  (r_4-east) -- node {\midarrow} (ct3-west);
\draw [connection]  (r_3-east) -- node {\midarrow} (ct2-west);
\draw [connection]  (r_2-east) -- node {\midarrow} (ct1-west);
\draw [connection]  (r_1-east) -- node {\midarrow} (ct0-west);
""",

    # arrows on RHS between convconvrelu and upconv
    # only keeping arrow to softmax
    r"""
\draw [connection]  (ct0-east) -- node {\midarrow} (out-west);
""",
    # arrow between LHS and concat symbol
    r"""
\draw [copyconnection]  (lr7-east)  -- node {\copymidarrow} (cat7-west);
\draw [copyconnection]  (lr6-east)  -- node {\copymidarrow} (cat6-west);
\draw [copyconnection]  (lr5-east)  -- node {\copymidarrow} (cat5-west);
\draw [copyconnection]  (lr4-east)  -- node {\copymidarrow} (cat4-west);
\draw [copyconnection]  (lr3-east)  -- node {\copymidarrow} (cat3-west);
\draw [copyconnection]  (lr2-east)  -- node {\copymidarrow} (cat2-west);
\draw [copyconnection]  (lr1-east)  -- node {\copymidarrow} (cat1-west);
""",
    # arrow from concat to RHS
    r"""
\draw [copyconnection]  (cat7-east)  -- node {\copymidarrow} (r_7-west);
\draw [copyconnection]  (cat6-east)  -- node {\copymidarrow} (r_6-west);
\draw [copyconnection]  (cat5-east)  -- node {\copymidarrow} (r_5-west);
\draw [copyconnection]  (cat4-east)  -- node {\copymidarrow} (r_4-west);
\draw [copyconnection]  (cat3-east)  -- node {\copymidarrow} (r_3-west);
\draw [copyconnection]  (cat2-east)  -- node {\copymidarrow} (r_2-west);
\draw [copyconnection]  (cat1-east)  -- node {\copymidarrow} (r_1-west);
""",
    # arrow from below to concat
    r"""
\draw [copyconnection]  (drop_7-north)  -- node {\copymidarrow} (cat7-south);
\draw [copyconnection]  (drop_6-north)  -- node {\copymidarrow} (cat6-south);
\draw [copyconnection]  (drop_5-north)  -- node {\copymidarrow} (cat5-south);
\draw [copyconnection]  (bn_4-north)  -- node {\copymidarrow} (cat4-south);
\draw [copyconnection]  (bn_3-north)  -- node {\copymidarrow} (cat3-south);
\draw [copyconnection]  (bn_2-north)  -- node {\copymidarrow} (cat2-south);
\draw [copyconnection]  (bn_1-north)  -- node {\copymidarrow} (cat1-south);
""",
    
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()