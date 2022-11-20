from argparse import ArgumentParser
from config_parser import get_config
from processing import Processor
from projector_initial import Projector
from segmentor import Segmentor


def main(args):
    general, segment = get_config((args.cam_conf, args.seg_conf))

    prc = Processor()
    prj = Projector(general)
    segmentor = Segmentor(segment)

    image = prc.open(args.image)
    undistorted = prj.remove_optical_distortion(image)

    segmented = segmentor.segment_image(undistorted)

    bird_eye_view = prj.make_projection(segmented)
    prc.save(bird_eye_view, args.result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cam_conf', help='Config file location',
                        default='config.yaml')
    parser.add_argument('--seg_conf', help='Config file location',
                        default='segmentation.yaml')
    parser.add_argument('--image', help='Image location')
    parser.add_argument('--result', help='Loacation to save results',
                        default='result')
    arguments = parser.parse_args()
    main(arguments)
