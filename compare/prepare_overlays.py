import os
import subprocess


def process_overlay(name, x, y, w, h, pdfpath, color='red'):
    _process_overlay(name, x, y, w, h, pdfpath, color=color)
    _process_overlay(name+'_replicate', x, y, w, h, pdfpath, color=color)

def _process_overlay(name, x, y, w, h, pdfpath, color='red'):
    filepath = os.path.join(pdfpath, name + '.pdf')
    print("Converting {} ...".format(filepath))

    png_name = 'overlays/{}.png'.format(name)
    cmds = [['convert', '-density', '300', filepath, png_name],
            ['convert', '-channel', 'RGB', png_name, '+level-colors', '{},'.format(color), png_name],
            ['convert', '-size', '1000x1000', 'xc:transparent', 'overlays/background.png'],
            ['composite', '-geometry', '{}x{}!+{}+{}'.format(w, h, x, y),
             png_name, 'overlays/background.png', png_name]]
    for cmd in cmds:
        subprocess.run(cmd)
    os.remove('overlays/background.png')


if __name__ == '__main__':
    pdfpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../figures/pdfs/'))
    process_overlay('Figure_4A', 145,  39, 785, 797, pdfpath)
    process_overlay('Figure_4C',  84,  32, 788, 870, pdfpath)
    process_overlay('Figure_4D',  58,  37, 866, 799, pdfpath)
    process_overlay('Figure_4E', 149,  55, 785, 798, pdfpath)
    process_overlay('Figure_4G',  88,  48, 789, 873, pdfpath)
    process_overlay('Figure_4H',  72, -38, 843, 899, pdfpath)
    process_overlay('Figure_4I', 149,  69, 785, 798, pdfpath)
    process_overlay('Figure_4K',  88,  66, 789, 870, pdfpath)
    process_overlay('Figure_4L',  72,  70, 769, 799, pdfpath)
