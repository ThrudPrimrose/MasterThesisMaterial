import subprocess

for i in range(4):
    offset = "0." + str(2*i)
    with open(f"row-wise-iter-{i}.tex", 'w') as file:

        template=f"""
\\documentclass[tikz,border=0.1cm,usenames,dvipsnames,convert={{outext=.svg}}]{{standalone}}
\\usepackage{{tikz,tikz-3dplot}}
\\usepackage{{amsmath,amsthm}}
\\usepackage{{amssymb}}
\\usepackage{{amsfonts}}
\\usepackage{{times}}
\\usepackage{{svg}}

\\usetikzlibrary{{positioning,arrows.meta,quotes}}
\\usetikzlibrary{{shapes,snakes}}
\\usetikzlibrary{{bayesnet}}
\\tikzset{{>=latex}}

\\begin{{document}}
\\begin{{tikzpicture}}
    % A
    \\draw [thick,draw=LimeGreen!40!black] (.8,0) rectangle (1.6,1.2);
    \\filldraw [fill=LimeGreen!10!white,draw=LimeGreen!40!black] (.8,0) rectangle (1.6,1.2);
    \\draw [step=0.4/2, very thin, color=gray] (.8,0) grid (1.6,1.2);
    
    \\draw (1.0,-0.2) node {{{{\\color{{black}}\\tiny{{$A\\in\\mathbb{{R}}^{{\\tiny{{M \\times K}}}}$}}}}}};

    % Code Row-wise, iter 1
    \\draw [thick,draw=BlueViolet!40!black] (.8+{offset},1.0) rectangle (1.0+{offset},1.2);
    \\filldraw [fill=BlueViolet!10!white,draw=BlueViolet!40!black] (.8+{offset},1.0) rectangle (1.0+{offset},1.2);
    \\draw [step=0.4/2, very thin, color=gray] (.8+{offset},1.0) grid (1.0+{offset},1.2);
    \\draw[thin, draw=Gray, ->=, >=stealth'] (0.85, 1.3) -- (1.55, 1.3);

    %Code Inner Prod
    %\\draw[thin, draw=Red] (.85, 1.1) -- (1.55, 1.1);
    %\\draw[thin, draw=Gray, ->=, >=stealth'] (0.85, 1.3) -- (1.55, 1.3);

    % B
    \\draw [thick,draw=Cerulean!40!black] (2,1.6) rectangle (2.8,2.4);
    \\filldraw [fill=Cerulean!10!white,draw=Cerulean!40!black] (2,1.6) rectangle (2.8,2.4);
    \\draw [step=0.4/2, very thin, color=gray] (2, 1.6) grid (2.8,2.4);
    
    \\draw (2.4 , 2.75) node {{{{\\color{{black}}\\tiny{{$B\\in\\mathbb{{R}}^{{\\tiny{{K \\times N}}}}$}}}}}};

    % Code Row-wise, iter 1
    \\draw [thick,draw=BlueViolet!40!black] (2,2.2-{offset}) rectangle (2.8,2.4-{offset});
    \\filldraw [fill=BlueViolet!10!white,draw=BlueViolet!40!black] (2,2.2-{offset}) rectangle (2.8,2.4-{offset});
    \\draw [step=0.4/2, very thin, color=gray] (2, 2.2-{offset}) grid (2.8,2.4-{offset});
    \\draw[thin, draw=Gray, ->=, >=stealth'] (2.05, 2.5) -- (2.75, 2.5);

    %Code Inner Prod
    %\\draw[thin, draw=Red] (2.1, 2.35) -- (2.1, 1.65);
    %\\draw[thin, draw=Gray, ->=] (1.9, 2.35) -- (1.9, 1.65);

    % C
    \\draw [thick,draw=BrickRed!40!black] (2,0) rectangle (2.8,1.2);
    \\filldraw [fill=BrickRed!10!white,draw=BrickRed!40!black] (2,0) rectangle (2.8,1.2);
    \\draw [step=0.4/2, very thin, color=gray] (2, 0) grid (2.8,1.2);


    % Code Row-wise, iter 1
    \\draw [thick,draw=BlueViolet!{40+float(offset)*40}!black] (2,1.0) rectangle (2.8,1.2);
    \\filldraw [fill=BlueViolet!{10+float(offset)*40}!white,draw=BlueViolet!{40+float(offset)*40}!black] (2,1.0) rectangle (2.8,1.2);
    \\draw [step=0.4/2, very thin, color=gray] (2, 1.0) grid (2.8,1.2);

    %Code Inner Prod
    %\\draw[thin, draw=Red] (2.05, 1.1) -- (2.15, 1.1);
    %\\draw[thin, draw=Red] (2.1, 1.15) -- (2.1, 1.05);

    \\draw (2.5 , -0.2) node {{{{\\color{{black}}\\tiny{{$C\\in\\mathbb{{R}}^{{\\tiny{{M \\times N}}}}$}}}}}};
\\end{{tikzpicture}}
\\end{{document}}
"""
        file.write(template)

for i in range(4):

    command = f"pdflatex row-wise-iter-{i}.tex"
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    command = f"inkscape -l row-wise-iter-{i}.pdf --export-filename=row-wise-iter-{i}_svg.svg"    
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
