import os
import copy
import subprocess


def process_overlay(name_xywhs, pdfpath, color='red'):
    _process_overlay(name_xywhs, pdfpath, color=color)
    name_xywhs = copy.deepcopy(name_xywhs)
    for name_xywh in name_xywhs:
        name_xywh[0] = name_xywh[0] + '_replicate'
    _process_overlay(name_xywhs, pdfpath, color=color)

def _process_overlay(name_xywh, pdfpath, color='red'):
    background_png = 'overlays/background.png'

    subprocess.run(['convert', '-size', '1000x1000', 'xc:transparent', 'overlays/background.png'])
    for name, x, y, w, h in name_xywh:
        filepath = os.path.join(pdfpath, name + '.pdf')
        print("Converting {} ...".format(filepath))

        png_name = 'overlays/{}.png'.format(name)
        cmds = [['convert', '-density', '300', filepath, png_name],
            # ['convert', '-channel', 'RGB', png_name, '+level-colors', '{},'.format(color), png_name],
                ['composite', '-geometry', '{}x{}!+{}+{}'.format(w, h, x, y),
                     png_name, background_png, background_png]]
        for cmd in cmds:
            subprocess.run(cmd)
        os.remove(png_name)


    output_name = 'overlays/{}.png'.format(name_xywh[0][0])
    os.rename('overlays/background.png', output_name)


if __name__ == '__main__':
    pdfpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../figures/pdfs/'))
    # process_overlay((('Figure_4A', 145,  39, 785, 797),), pdfpath)
    # process_overlay((('Figure_4C',  84,  32, 788, 870),), pdfpath)
    # process_overlay((('Figure_4D',  58,  37, 866, 799),), pdfpath)
    # process_overlay((('Figure_4E', 149,  55, 785, 798),), pdfpath)
    # process_overlay((('Figure_4G',  88,  48, 789, 873),), pdfpath)
    # process_overlay((('Figure_4H',  72, -38, 843, 899),), pdfpath)
    # process_overlay((('Figure_4I', 149,  69, 785, 798),), pdfpath)
    # process_overlay((('Figure_4K',  88,  66, 789, 870),), pdfpath)
    # process_overlay((('Figure_4L',  72,  70, 769, 799),), pdfpath)
    #
    # process_overlay((('Figure_5B',  89,   7,1019, 890),), pdfpath)
    # process_overlay((('Figure_5C',  90,  16,1017, 891),), pdfpath)
    # process_overlay((('Figure_5D', 106,  15,1016, 892),), pdfpath)
    #
    # process_overlay((('Figure_6A', 154,  54, 826, 841),), pdfpath)
    # process_overlay((('Figure_6C', 127,  55, 826, 840),), pdfpath)
    # process_overlay((('Figure_6E', 141,  87, 826, 839),), pdfpath)
    # process_overlay((('Figure_6G', 102,  88, 826, 838),), pdfpath)
    # process_overlay((('Figure_6I', 140,  73, 827, 841),), pdfpath)
    # process_overlay((('Figure_6K', 102,  73, 826, 841),), pdfpath)

    process_overlay((['Figure_7C',       120.5, 75, 807.5, 889],
                     ['Figure_7C_inset', 140  , 35, 400.5, 405]),
                    pdfpath)
