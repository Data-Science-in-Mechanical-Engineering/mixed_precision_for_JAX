\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dsme}[2025/06/02 v2.0 DSME Document Class]

%%======== Package options and base class ========
\RequirePackage{xifthen}
\RequirePackage{kvoptions}

% Define class options
\newboolean{nonumbering}
\newboolean{logo}
\newboolean{copyright}
\newboolean{compact}
\newboolean{draft}

\DeclareOption{nonumbering}{\setboolean{nonumbering}{true}}
\DeclareOption{logo}{\setboolean{logo}{true}}
\DeclareOption{copyright}{\setboolean{copyright}{true}}
\DeclareOption{compact}{\setboolean{compact}{true}}
\DeclareOption{draft}{\setboolean{draft}{true}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}

\ProcessOptions\relax

% Load base class
\LoadClass{extarticle}

%%======== Required packages ========
% Essential packages
\RequirePackage{cmap}  % Moved before fontenc to avoid warning
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[english]{babel}

% Mathematics - Note: Users should load amsmath, amsfonts, amssymb, amsthm, mathtools after this class
% These are commented out to avoid conflicts with user's explicit loading
% \RequirePackage{amsmath, amsfonts, amssymb, amsthm}
% \RequirePackage{mathtools}

% Graphics and colors - Note: users load xcolor and graphicx separately
\RequirePackage{tikz}
\RequirePackage{xcolor}

% Typography and fonts - Minimal setup to work with user's mathdesign+charter
% Note: Users load mathdesign[charter] which provides both text and math fonts
\RequirePackage[scaled=1.1]{zlmtt}  % Typewriter font only
\AtEndOfClass{\RequirePackage[tracking=smallcaps,expansion=false]{microtype}}

% Note: Template expects users to load:
\usepackage[charter]{mathdesign}  % Provides both Charter text and matching math
\usepackage{eucal}                % For elegant script fonts

% Tables and formatting
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{colortbl}
\RequirePackage{array}

% Lists and spacing
\RequirePackage{enumitem}
\setlist[itemize]{noitemsep, topsep=0.5ex}
\setlist[enumerate]{noitemsep, topsep=0.5ex}
\setlist[description]{style=nextline, leftmargin=0pt}

% Links and references - Note: users load hyperref separately
\RequirePackage{url}
\RequirePackage[colorlinks=true, allcolors=RWTHRot100, breaklinks=true]{hyperref}
% \RequirePackage[capitalize, noabbrev]{cleveref}

% Headers, footers, and page layout
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{changepage}
\RequirePackage{zref-totpages}

% Date and time
\RequirePackage{datetime2} 
\DTMsetdatestyle{iso}  

% Bibliography and citations
% Note: Users should load natbib after this class if desired
% \RequirePackage[authoryear, sort&compress, round]{natbib}
\RequirePackage{csquotes}

% Title formatting
\RequirePackage[explicit]{titlesec}
\RequirePackage{authblk}

% Captions and floats
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{float}
\RequirePackage{tcolorbox}
\RequirePackage{minted}
\usemintedstyle{tango}

% Additional utilities
\RequirePackage{textcomp}
\RequirePackage{needspace}
\RequirePackage{etoolbox}
\RequirePackage{calc}
\RequirePackage{fontawesome5}

%%======== Page geometry and spacing ========
\RequirePackage[
    left=2.5cm,
    right=2.5cm,
    top=3cm,
    bottom=3cm,
    headheight=14pt,
    headsep=20pt,
    footskip=30pt,
    a4paper
]{geometry}

% Adjust spacing for compact option
\ifthenelse{\boolean{compact}}{%
    \geometry{left=2cm, right=2cm, top=2.5cm, bottom=2.5cm}
    \setlength{\parskip}{0.3\baselineskip}
}{%
    \setlength{\parskip}{0.5\baselineskip}
}

\setlength{\columnsep}{20pt}
\setlength{\parindent}{0pt}

%%======== Color scheme ========
\definecolor{dsmeblue}{RGB}{0, 84, 159}
\definecolor{dsmegray}{RGB}{88, 88, 90}
\definecolor{dsmelight}{RGB}{240, 240, 240}
\definecolor{dsmeaccent}{RGB}{227, 114, 34}
%%% RWTH Colors
% Hausfarbe
\definecolor{RWTHBlau100}{RGB}{0,84,159}	% Logo primary color
\definecolor{RWTHBlau75}{RGB}{64,127,183}
\definecolor{RWTHBlau50}{RGB}{142,186,229}	% Logo secondary color
\definecolor{RWTHBlau25}{RGB}{199,221,242}
\definecolor{RWTHBlau10}{RGB}{232,241,250}

% Sekundärfarben
% Schwarz
\definecolor{RWTHSchwarz100}{RGB}{0,0,0}
\definecolor{RWTHSchwarz75}{RGB}{100,101,103}
\definecolor{RWTHSchwarz50}{RGB}{156,158,159}
\definecolor{RWTHSchwarz25}{RGB}{207,209,210}
\definecolor{RWTHSchwarz10}{RGB}{236,237,237}

% Magenta
\definecolor{RWTHMagenta100}{RGB}{227,0,102}
\definecolor{RWTHMagenta75}{RGB}{233,96,136}
\definecolor{RWTHMagenta50}{RGB}{241,158,177}
\definecolor{RWTHMagenta25}{RGB}{249,210,218}
\definecolor{RWTHMagenta10}{RGB}{253,238,240}

% Gelb
\definecolor{RWTHGelb100}{RGB}{255,237,0}
\definecolor{RWTHGelb75}{RGB}{255,240,85}
\definecolor{RWTHGelb50}{RGB}{255,245,155}
\definecolor{RWTHGelb25}{RGB}{255,250,209}
\definecolor{RWTHGelb10}{RGB}{255,253,238}

% Petrol
\definecolor{RWTHPetrol100}{RGB}{0,97,101}
\definecolor{RWTHPetrol75}{RGB}{45,127,131}
\definecolor{RWTHPetrol50}{RGB}{125,164,167}
\definecolor{RWTHPetrol25}{RGB}{191,208,209}
\definecolor{RWTHPetrol10}{RGB}{230,236,236}

% Türkis
\definecolor{RWTHTuerkis100}{RGB}{0,152,161}
\definecolor{RWTHTuerkis75}{RGB}{0,177,183}
\definecolor{RWTHTuerkis50}{RGB}{137,204,207}
\definecolor{RWTHTuerkis25}{RGB}{202,231,231}
\definecolor{RWTHTuerkis10}{RGB}{235,246,246}

% Grün
\definecolor{RWTHGruen100}{RGB}{87,171,39}
\definecolor{RWTHGruen75}{RGB}{141,192,96}
\definecolor{RWTHGruen50}{RGB}{184,214,152}
\definecolor{RWTHGruen25}{RGB}{221,235,206}
\definecolor{RWTHGruen10}{RGB}{242,247,236}

% Maigrün
\definecolor{RWTHMaigruen100}{RGB}{189,205,0}
\definecolor{RWTHMaigruen75}{RGB}{208,217,92}
\definecolor{RWTHMaigruen50}{RGB}{224,230,154}
\definecolor{RWTHMaigruen25}{RGB}{240,243,208}
\definecolor{RWTHMaigruen10}{RGB}{249,250,237}	

% Orange
\definecolor{RWTHOrange100}{RGB}{246,168,0}
\definecolor{RWTHOrange75}{RGB}{250,190,80 }
\definecolor{RWTHOrange50}{RGB}{253,212,143}
\definecolor{RWTHOrange25}{RGB}{254,234,201}
\definecolor{RWTHOrange10}{RGB}{255,247,234}	

% Rot
\definecolor{RWTHRot100}{RGB}{204,7,30}
\definecolor{RWTHRot75}{RGB}{216,92,65}
\definecolor{RWTHRot50}{RGB}{230,150,121}
\definecolor{RWTHRot25}{RGB}{243,205,187}
\definecolor{RWTHRot10}{RGB}{250,235,227}	

% Bordeaux
\definecolor{RWTHBordeaux100}{RGB}{161,16,53}
\definecolor{RWTHBordeaux75}{RGB}{182,82,86}
\definecolor{RWTHBordeaux50}{RGB}{205,139,135}
\definecolor{RWTHBordeaux25}{RGB}{229,197,192}
\definecolor{RWTHBordeaux10}{RGB}{245,232,229}

% Violett
\definecolor{RWTHViolett100}{RGB}{97,33,88}
\definecolor{RWTHViolett75}{RGB}{131,78,117}
\definecolor{RWTHViolett50}{RGB}{168,133,158}
\definecolor{RWTHViolett25}{RGB}{210,192,205}
\definecolor{RWTHViolett10}{RGB}{237,229,234}	

% Lila
\definecolor{RWTHLila100}{RGB}{122,111,172}
\definecolor{RWTHLila75}{RGB}{155,145,193}
\definecolor{RWTHLila50}{RGB}{188,181,215}
\definecolor{RWTHLila25}{RGB}{222,218,235}
\definecolor{RWTHLila10}{RGB}{242,240,247}

%%======== Custom font commands ========
\newcommand{\titlefont}{\color{dsmeblue}\bfseries\fontsize{22}{26}\selectfont}
\newcommand{\authorfont}{\normalfont\fontsize{11}{13}\selectfont}
\newcommand{\affiliationfont}{\normalfont\itshape\fontsize{9}{11}\selectfont}
\newcommand{\datefont}{\normalfont\itshape\fontsize{9}{11}\selectfont}
\newcommand{\headerfont}{\normalfont\fontsize{9}{11}\selectfont}
\newcommand{\footerfont}{\normalfont\fontsize{8}{10}\selectfont}
\newcommand{\abstractfont}{\normalfont\fontsize{10}{12}\selectfont}
\newcommand{\keywordsfont}{\itshape\fontsize{10}{12}\selectfont}
\newcommand{\sectionfont}{\color{dsmeblue}\bfseries\fontsize{14}{16}\selectfont}
\newcommand{\subsectionfont}{\color{black}\bfseries\fontsize{12}{14}\selectfont}
\newcommand{\codeurl}[1]{\def\@codeurl{#1}}
\newcommand{\codeurlshort}[1]{\def\@codeurlshort{#1}}
\newcommand{\codelabel}[1]{\def\@codelabel{#1}}
\newcommand{\videourl}[1]{\def\@videourl{#1}}
\newcommand{\videolabel}[1]{\def\@videolabel{#1}}

%%======== Document metadata commands ========
\newcommand{\keywords}[1]{\def\@keywords{#1}}
\newcommand{\correspondingauthor}[1]{\def\@correspondingauthor{#1}}
\newcommand{\paperurl}[1]{\def\@paperurl{#1}}
\newcommand{\reportnumber}[1]{\def\@reportnumber{#1}}
\newcommand{\institution}[1]{\def\@institution{#1}}
\newcommand{\department}[1]{\def\@department{#1}}

% Set defaults
\def\@keywords{}
\def\@correspondingauthor{}
\def\@paperurl{}
\def\@reportnumber{}
\def\@institution{Institute for Data Science in Mechanical Engineering (DSME), RWTH Aachen University}
\def\@department{DSME}
\def\@codeurl{}
\def\@codeurlshort{}
\def\@codelabel{Code:}
\def\@videourl{}
\def\@videolabel{Video:}

%%======== Author and affiliation setup ========
\setcounter{Maxaffil}{0}
\renewcommand\Authfont{\authorfont\bfseries}
\renewcommand\Affilfont{\affiliationfont}
\renewcommand\Authands{ and }
\setlength{\affilsep}{0.5em}

%%======== Header and footer setup ========
\pagestyle{fancy}
\fancyhf{} % Clear all header and footer fields

\renewcommand{\headrulewidth}{0pt}
\newcommand{\shortauthors}{Anonymous et al.}

% First page style
\fancypagestyle{firststyle}{%
    \fancyhf{}
    \fancyfoot[L]{%
        \footerfont
        \ifx\@correspondingauthor\@empty\else
            \color{dsmegray}Corresponding author: \@correspondingauthor \\
        \fi
        \ifthenelse{\boolean{copyright}}{%
            \color{dsmegray}\textcopyright\ \the\year\ \@institution.\\ All rights reserved.%
        }{}%
    }
    \fancyfoot[R]{%
    \color{dsmegray}
        \ifthenelse{\boolean{logo}}{%
            \raisebox{-13pt}[0pt][0pt]{\includegraphics[height=20pt]{assets/rwth_data_science_im_maschinenbau_bild_rgb.png}}%
            \ifx\@reportnumber\@empty\else
                \\ \footerfont Report No. \@reportnumber
            \fi
        }{%
        }%
    }
    % \renewcommand{\headrulewidth}{0.5pt}
    % \renewcommand{\footrulewidth}{0.5pt}
}

% Subsequent pages style
\fancyhead[R]{\footerfont \shortauthors}

\fancyfoot[L]{\color{dsmegray}\footerfont\textcopyright\ \the\year\ \@department}
\fancyfoot[C]{\color{dsmegray}\footerfont Page \thepage\ of \ztotpages}
\fancyfoot[R]{%
    \footerfont
    \ifx\@reportnumber\@empty\else
        \@reportnumber
    \fi
}

% \renewcommand{\headrulewidth}{0.5pt}
% \renewcommand{\footrulewidth}{0.5pt}
% \renewcommand{\headrule}{\color{dsmegray}\hrule width\headwidth height\headrulewidth}
% \renewcommand{\footrule}{\color{dsmegray}\hrule width\headwidth height\footrulewidth}

%%======== Section numbering ========
\setcounter{secnumdepth}{3}
\ifthenelse{\boolean{nonumbering}}{\setcounter{secnumdepth}{0}}{}

%%======== Title formatting ========
\titleformat{\section}
    {\sectionfont}
    {\thesection}
    {1em}
    {#1}

\titleformat{\subsection}
    {\subsectionfont}
    {\thesubsection}
    {1em}
    {#1}

\titleformat{\subsubsection}
    {\bfseries}
    {\thesubsubsection}
    {1em}
    {#1}

\titlespacing*{\section}{0pt}{3ex plus 1ex minus 0.2ex}{1.5ex plus 0.2ex}
\titlespacing*{\subsection}{0pt}{2.5ex plus 1ex minus 0.2ex}{1ex plus 0.2ex}
\titlespacing*{\subsubsection}{0pt}{2ex plus 0.5ex minus 0.2ex}{0.5ex plus 0.2ex}

%%======== Enhanced Abstract environment with code link ========
\renewenvironment{abstract}{%
    \begin{center}
        \begin{minipage}{0.85\textwidth}
            \centering
            \abstractfont
            \noindent
            \begin{adjustwidth}{0pt}{0pt}
}{%          
            \end{adjustwidth}
            \centering
            \ifx\@keywords\@empty\else
                \vskip 1em
                \noindent\textbf{Keywords:} \keywordsfont\@keywords
            \fi
            \ifx\@videourl\@empty\else
                \vskip 1em
                \noindent\textbf{\@videolabel} \href{\@videourl}{\color{blue}\url{\@videourl}}
            \fi
            \ifx\@codeurl\@empty\else
                \vskip 1.5em
                \noindent\textbf{\@codelabel} \href{\@codeurl}{\@codeurlshort} 
            \fi
        \end{minipage}
    \end{center}
    \vskip 2em
}
%%======== Enhanced maketitle ========
\renewcommand{\maketitle}{%
    \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
        \hb@xt@1.8em{\hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \@maketitle
    \thispagestyle{firststyle}
    \@thanks
    \endgroup
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\title\relax
    \global\let\author\relax
    \global\let\date\relax
    \global\let\and\relax
}

\def\@maketitle{%
    \newpage
    \vspace*{-0.8cm}
    \begin{center}%
        \let\footnote\thanks
        {\titlefont \@title \par}%
        \vskip 1.5em%
        {\large
            \lineskip .5em%
            \begin{tabular}[t]{c}%
                \@author
            \end{tabular}\par}%
    \end{center}%
    \par
    \vskip 1.5em
}

%%======== Enhanced captions ========
\captionsetup{
    labelfont={bf,color=dsmeblue},
    textfont={small},
    labelsep=period,
    format=hang,
    margin=10pt
}

\captionsetup[sub]{
    labelfont={bf,color=dsmeblue},
    textfont={footnotesize},
    format=plain,
    labelformat=parens,
}

%%======== Theorem environments ========
% Completely skip theorem definitions - users will define their own with amsthm/thmtools

%%======== Custom environments ========
% Highlight box for important information
\newtcolorbox{highlight}[1][Important]{%
  colback=RWTHOrange10!50!white,
  colframe=dsmeaccent,
  fonttitle=\bfseries,
  coltitle=white,
  title=#1,
  left=8pt, right=8pt, top=6pt, bottom=6pt,
}

% Code listing environment
\newtcolorbox{pbox}[2][]{%
  colback=RWTHBlau10!10!white,
  colframe=RWTHBlau75,
  coltitle=white,
  toptitle=0.125cm,
  bottomtitle=0.125cm,
  fonttitle=\bfseries,
  title=#2,
  #1
}

\newtcolorbox{plainbox}[1][]{%
  colback=RWTHBlau10!10!white,
  colframe=RWTHBlau100,
  fonttitle=\bfseries,
  #1
}

%%======== Draft mode features ========
\ifthenelse{\boolean{draft}}{%
    \RequirePackage{draftwatermark}
    \SetWatermarkText{DRAFT}
    \SetWatermarkScale{3}
    \SetWatermarkLightness{0.9}
    \RequirePackage[displaymath, mathlines]{lineno}
    \linenumbers
    \renewcommand{\footerfont}{\color{red}\normalfont\fontsize{8}{10}\selectfont}
}{}

%%======== Bibliography setup ========
% Bibliography formatting (compatible with natbib)
\ifthenelse{\boolean{compact}}{%
    % For compact mode, reduce bibliography spacing
    \AtBeginDocument{%
        \@ifpackageloaded{natbib}{%
            \setlength{\bibsep}{0.3\baselineskip}
        }{}
    }
}{%
    % Normal bibliography spacing
    \AtBeginDocument{%
        \@ifpackageloaded{natbib}{%
            \setlength{\bibsep}{0.5\baselineskip}
        }{}
    }
}

% Rename bibliography section
\AtBeginDocument{%
    \@ifpackageloaded{natbib}{%
        \renewcommand{\bibname}{References}
        \renewcommand{\refname}{References}
    }{}
}

%%======== Final adjustments ========
% Improve spacing around equations
\AtBeginDocument{%
    \setlength{\abovedisplayskip}{1em plus 0.2em minus 0.2em}
    \setlength{\belowdisplayskip}{1em plus 0.2em minus 0.2em}
    \setlength{\abovedisplayshortskip}{0.5em plus 0.2em minus 0.2em}
    \setlength{\belowdisplayshortskip}{0.5em plus 0.2em minus 0.2em}
}

% Better float placement
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.7}
\renewcommand{\dblfloatpagefraction}{0.7}

\endinput