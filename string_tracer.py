"""Image String Tracer.

Usage: string_tracer.py <imagefile>
"""

__author__ = "Dmitry Andreev"
__credits__ = ["Dmitry Andreev"]
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License."
__version__ = "0.0.1"

import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import random
import time
import numpy as np
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


ANCHOR_COUNT = 200
MAX_TRACE_COUNT = 4000


def map_line(val):

    return 1.0 - val * 0.3


def draw_line(buf, last, curr):

    rr, cc, val = line_aa(int(last[1]), int(last[0]), int(curr[1]), int(curr[0]))
    buf[rr, cc] *= map_line(val)

    return buf


def line_error(item):

    i, last_idx, anchors, last_val, target_val, j = item

    rr, cc, val = line_aa(
        int(anchors[last_idx][1]), int(anchors[last_idx][0]),
        int(anchors[i][1]), int(anchors[i][0]))

    err = np.sum(
        + (pow(last_val[rr, cc] * map_line(val), 0.5) - target_val[rr, cc])**2
        - (pow(last_val[rr, cc], 0.5) - target_val[rr, cc])**2
        )

    return (err, i)


def main():

    try:
        target_path = sys.argv[1]

    except:
        print("Usage: string_tracer.py <imagefile>")
        return

    sz = 512
    trace_list = []

    pygame.init()

    target_img = pygame.image.load(target_path)

    # Downscale and crop to 512 x 512
    target_size = target_img.get_size()
    target_img = pygame.transform.smoothscale(target_img, (
        (target_size[0] + 1) * sz // min(target_size),
        (target_size[1] + 1) * sz // min(target_size)))

    cx = (target_img.get_size()[0] - sz) // 2
    cy = (target_img.get_size()[1] - sz) // 2

    target_img = pygame.transform.grayscale(target_img)
    target_arr = pygame.surfarray.array3d(target_img)
    target_arr = target_arr[cx::,cy::]
    target_val = pow(target_arr[cx::,cy::,1].astype(np.float32) / 255.0, 0.5)
    target_val -= np.mean(target_val)

    mx, my = np.mgrid[-1:1:1/float(sz * 0.5), -1:1:1/float(sz * 0.5)]
    mask = (mx*mx + my*my) <= 0.96
    mask_total = float(np.sum(mask))

    N = ANCHOR_COUNT

    anchors = []
    o = sz * 0.5

    for i in range(N):
        f = 2.0 * np.pi * float(i) / float(N)
        anchors.append((
            o + 0.98 * o * np.sin(f) + (random.random() * 2.0 - 1.0) * 0.25,
            o + 0.98 * o * np.cos(f) + (random.random() * 2.0 - 1.0) * 0.25))

    img = pygame.Surface((sz, sz))
    screen = pygame.display.set_mode((sz, sz), 0, 32)

    last_idx = 0
    last_idx2 = 0
    line_count = 0

    pool = ThreadPool()
    last_val = np.ones((sz, sz), dtype=np.float32)

    anchor_turns = np.zeros(len(anchors), dtype=np.int32)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        if line_count < MAX_TRACE_COUNT:

            jobs = []

            for i, a in enumerate(anchors):

                # Don't visit previous anchors right away
                idist1 = min(abs(i - last_idx), abs(i + len(anchors) - last_idx))
                idist2 = min(abs(i - last_idx2), abs(i + len(anchors) - last_idx2))

                if idist1 < (N // 16) or idist2 < (N // 16):
                    continue

                jobs.append((i, last_idx, anchors, last_val, target_val, -1))

            errs = pool.map(line_error, jobs)
            min_err, min_idx = min(errs)

            if min_idx >= 0:
                i = min_idx
                last_val = draw_line(last_val, anchors[last_idx], anchors[i])
                anchor_turns[i] += 1

                trace_list.append((line_count + 1, last_idx, i, anchors[last_idx][1], anchors[last_idx][0], anchors[i][1], anchors[i][0]))

                # Jitter anchor points to avoid aliasing
                f = 2.0 * np.pi * float(i) / float(N)
                anchors[i] = (
                    o + 0.98 * o * np.sin(f) + (random.random() * 2.0 - 1.0) * 2.0,
                    o + 0.98 * o * np.cos(f) + (random.random() * 2.0 - 1.0) * 2.0)

                last_idx2 = last_idx
                last_idx = i

                print(line_count, abs(min_err))
                line_count += 1

        if line_count > MAX_TRACE_COUNT:
            time.sleep(0.05)

        # Update every 8th frame
        if line_count % 8 and line_count != MAX_TRACE_COUNT:
            continue

        # Apply Tone mapping and gamma correction
        m = pygame.surfarray.pixels3d(img)
        avg = np.sum(last_val, where=mask) / mask_total
        v = 0.3 * last_val / avg
        v = pow(np.minimum(1.0, v * 1.5 / (1.0 + v)), 0.4545)
        v = pow(v * 0.5 + 0.5 * v * v * (3.0 - 2.0 * v), 1.0) # smoothstep (contrast)
        m[:,:,0] = m[:,:,1] = m[:,:,2] = (v * 255.0).astype(np.uint8)
        del m

        if line_count == MAX_TRACE_COUNT:
            line_count += 1
            base_path = os.path.dirname(target_path)
            name, ext = os.path.splitext(os.path.basename(target_path))
            out_png_path = os.path.join(base_path, name) + ".traced.png"
            out_csv_path = os.path.join(base_path, name) + ".traced.csv"
            out_svg_path = os.path.join(base_path, name) + ".traced.svg"

            print('.saving "{}"'.format(out_png_path))
            pygame.image.save(img, out_png_path)

            with open(out_csv_path, 'wt') as f:
                print('.saving "{}"'.format(out_csv_path))
                f.write("Step,First Index,Second Index,First X,First Y,Second X,Second Y\n")
                f.write('\n'.join([','.join([str(v) for v in t]) for t in trace_list]))
                f.write('\n')

            with open(out_svg_path, 'wt') as f:
                print('.saving "{}"'.format(out_svg_path))

                f.write('<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">\n')
                f.write('<g stroke="black" stroke-width="0.09">\n')

                for t in trace_list:
                    f.write('<line x1="{}" y1="{}" x2="{}" y2="{}" />\n'.format(t[3], t[4], t[5], t[6]))

                f.write('</g></svg>\n')

        screen.blit(img, (0, 0))
        pygame.display.flip()


if __name__ == "__main__":

    main()
