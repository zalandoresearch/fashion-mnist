import sys
from collections import defaultdict

from typing import Dict, Any, Union


def parse_arg(v: str) -> Union[str, int, float, bool, None]:
    try:
        v = int(v)  # parse int parameter
    except ValueError:
        try:
            v = float(v)  # parse float parameter
        except ValueError:
            if len(v) == 0:
                # ignore it when the parameter is empty
                v = None
            elif v.lower() == 'true':  # parse boolean parameter
                v = True
            elif v.lower() == 'false':
                v = False
    return v


def get_args_request(args: Dict[str, str]) -> Dict[str, Any]:
    return {k: parse_arg(v) for k, v in args.items()}


def get_args_cli() -> Dict[str, Any]:
    d = defaultdict(list)
    if sys.argv[1:]:
        for k, v in ((k.lstrip('-'), v) for k, v in (a.split('=') for a in sys.argv[1:])):
            d[k].append(v)
        for k, v in d.items():
            parsed_v = [s for s in (parse_arg(vv) for vv in v) if s is not None]
            if len(parsed_v) > 1:
                d[k] = parsed_v
            if len(parsed_v) == 1:
                d[k] = parsed_v[0]
    return d
