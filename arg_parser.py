from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()

    parser.add_argument('--max_tokens', type=int,
                        default=1024, help='max_tokens')
    parser.add_argument('--temperature', type=float,
                        default=0.8, help='temperature')
    parser.add_argument('--top_k', type=int, default=60, help='top_k')
    parser.add_argument('--top_p', type=float, default=0.6, help='top_p')
    parser.add_argument('--repetition_penalty', type=float,
                        default=1.1, help='repetition_penalty')
    parser.add_argument('--stop', type=list, default=['</s>'], help='stop')
    
    return parser