
import os
import sys
import subprocess

import PyPDF2


class PDF:

    def __init__(self, filename):
        self._fp = open(filename, 'rb')
        self.reader = PyPDF2.PdfFileReader(self._fp)

    def extract_image(self, page_number, x, y, w, h, out_name, blank_params=None):
        print('Extracting {}.pdf'.format(out_name))
        assert w == h  # we want to enforce squares

        page = self.reader.getPage(page_number)

        #backing up
        intact_mediabox = page.mediaBox.lowerLeft, page.mediaBox.upperRight
        intact_cropbox  = page.cropBox.lowerLeft, page.cropBox.upperRight
        intact_trimbox  = page.trimBox.lowerLeft, page.trimBox.upperRight

        w_p, h_p = page.mediaBox.upperRight
        x_ll, y_ll = x, h_p - y
        x_ur, y_ur = (x + w), h_p - (y + h)
        page.mediaBox.lowerLeft  = x_ll, y_ll
        page.mediaBox.upperRight = x_ur, y_ur
        page.cropBox.lowerLeft   = x_ll, y_ll
        page.cropBox.upperRight  = x_ur, y_ur
        page.trimBox.lowerLeft   = x_ll, y_ll
        page.trimBox.upperRight  = x_ur, y_ur

        writer = PyPDF2.PdfFileWriter()
        writer.addPage(page)
        with open(out_name + '.pdf', 'wb') as fp:
            writer.write(fp)

        self.png_convert(out_name, blank_params=blank_params)

        #restoring
        page.mediaBox.lowerLeft, page.mediaBox.upperRight = intact_mediabox
        page.cropBox.lowerLeft, page.cropBox.upperRight = intact_cropbox
        page.trimBox.lowerLeft, page.trimBox.upperRight = intact_trimbox


    def png_convert(self, out_name, blank_params=None):
        subprocess.run(['convert', '-density', '600', '-resize', '1000x1000!',
                        out_name + '.pdf', out_name + '.png'])
        if blank_params is not None:
            self.blankout(out_name, blank_params)
        os.remove(out_name + '.pdf')

    def blankout(self, name, blank_params):
        x, y, w, h = blank_params
        subprocess.run(['convert', name + '.png', '-fill', 'white', '-draw',
                        'rectangle {},{} {},{}'.format(x, y, x+w-1, y+h-1),
                        name + '.png'])

    def close(self):
        self._fp.close()


if __name__ == '__main__':
    pdf = PDF('Rustichini2015.pdf')

    # pdf.extract_image( 7,  75,  82, 128, 128, 'originals/Figure_4A')
    # pdf.extract_image( 7, 320,  82, 128, 128, 'originals/Figure_4C', (900, 0, 100, 1000))
    # pdf.extract_image( 7, 432,  82, 128, 128, 'originals/Figure_4D')
    # pdf.extract_image( 7,  75, 222, 128, 128, 'originals/Figure_4E')
    # pdf.extract_image( 7, 320, 222, 128, 128, 'originals/Figure_4G', (900, 0, 100, 1000))
    # pdf.extract_image( 7, 432, 222, 128, 128, 'originals/Figure_4H')
    # pdf.extract_image( 7,  75, 360, 128, 128, 'originals/Figure_4I')
    # pdf.extract_image( 7, 320, 360, 128, 128, 'originals/Figure_4K', (900, 0, 100, 1000))
    # pdf.extract_image( 7, 432, 360, 128, 128, 'originals/Figure_4L')
    #
    # pdf.extract_image( 8, 410,  69, 168, 168, 'originals/Figure_5B')
    # pdf.extract_image( 8, 410, 245, 168, 168, 'originals/Figure_5C')
    # pdf.extract_image( 8, 235, 245, 168, 168, 'originals/Figure_5D')
    #
    # pdf.extract_image( 9,  69,  87, 128, 128, 'originals/Figure_6A')
    # pdf.extract_image( 9, 198,  87, 128, 128, 'originals/Figure_6B')
    # pdf.extract_image( 9, 325,  87, 128, 128, 'originals/Figure_6C')
    # pdf.extract_image( 9, 445,  87, 128, 128, 'originals/Figure_6D')
    # pdf.extract_image( 9,  69, 217, 128, 128, 'originals/Figure_6E')
    # pdf.extract_image( 9, 198, 217, 128, 128, 'originals/Figure_6F')
    # pdf.extract_image( 9, 325, 217, 128, 128, 'originals/Figure_6G')
    # pdf.extract_image( 9, 445, 217, 128, 128, 'originals/Figure_6H')
    # pdf.extract_image( 9,  69, 347, 128, 128, 'originals/Figure_6I')
    # pdf.extract_image( 9, 198, 347, 128, 128, 'originals/Figure_6J')
    # pdf.extract_image( 9, 325, 347, 128, 128, 'originals/Figure_6K')
    # pdf.extract_image( 9, 445, 347, 128, 128, 'originals/Figure_6L')

    pdf.extract_image(10, 415, 248, 165, 165, 'originals/Figure_7C')
    pdf.extract_image(10, 415, 424, 165, 165, 'originals/Figure_7E')

    pdf.close()
