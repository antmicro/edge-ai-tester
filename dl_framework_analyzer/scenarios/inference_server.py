import sys
import argparse

from dl_framework_analyzer.utils.class_loader import load_class
import dl_framework_analyzer.utils.logger as logger


def main(argv):
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'protocolcls',
        help='RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
    )
    parser.add_argument(
        'runtimecls',
        help='Runtime-based class with the implementation of model runtime'
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args, _ = parser.parse_known_args(argv[1:])

    protocolcls = load_class(args.protocolcls)
    runtimecls = load_class(args.runtimecls)

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            protocolcls.form_argparse()[0],
            runtimecls.form_argparse()[0],
        ]
    )

    args = parser.parse_args(argv[1:])

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    protocol = protocolcls.from_argparse(args)
    runtime = runtimecls.from_argparse(protocol, args)

    runtime.run_server()


if __name__ == '__main__':
    main(sys.argv)
