from typing import Type
from ia import IAI, load_ai, save_ai, GeneticAlgoByTime, GeneticAlgo
from environment import IEnvironment, WalkingEnv, ConfigurableAnimalEnv
from argparse import ArgumentError, ArgumentParser, Action, Namespace
from QLearning import QLearning
from hill_climbing import HillClimbing

def main(options: Namespace):
    AIClass: Type[IAI] = options.ai
    EnvClass: Type[IEnvironment] = options.env

    print("Algorithme:", AIClass.__name__)
    
    if options.load is not None:
        ai = load_ai(options.load)
    else:
        ai = AIClass(EnvClass, **options.ai_kwargs)

    
    if not options.test:
        ai.train(show=options.show, **options.mode_kwargs)
    
        if options.save is not None:
            save_ai(ai, options.save)

    else:
        ai.test(show=options.show, **options.mode_kwargs)



if __name__ == '__main__':
    class DictAction(Action):        

        def __call__(self, parser, namespace, values, option_string=None):            
            dct = dict()
            for kv in values:
                if kv.count('=') != 1:
                    raise ArgumentError(self, f"Invalid key-value pair argument: '{kv}'\n\nusage: {option_string} <key>=<value>")
                key, value = kv.replace(' ', '').split('=')
                try:
                    dct[key] = eval(value)
                except NameError and SyntaxError:
                    dct[key] = value
            setattr(namespace, self.dest, dct)

    class EvalAction(Action): 

        def __call__(self, parser, namespace, value, option_string=None):
            try:
                val = eval(value)
            except NameError:
                raise ArgumentError(self, f"Argument could not be resolved: '{value}'")
            setattr(namespace, self.dest, val)
    
    parser = ArgumentParser()
    parser.add_argument('-a', '--ai', type=str, dest='ai', action=EvalAction, default=GeneticAlgoByTime)#, choices=['GeneticAlgo', 'GeneticAlgoByTime'])
    parser.add_argument('-e', '--env', type=str, dest='env', action=EvalAction, default=WalkingEnv, choices=['WalkingEnv', 'ConfigurableAnimalEnv'])
    parser.add_argument('-l', '--load', type=str, dest='load', action='store', default=None)
    parser.add_argument('-s', '--save', type=str, dest='save', action='store', default=None)
    parser.add_argument('-r', '--show', dest='show', action='store_true', default=False)
    parser.add_argument('-ak', '--ai-kwargs', type=str, dest='ai_kwargs', action=DictAction, default={}, nargs='+')
    parser.add_argument('-tk', '--train-kwargs', '--test-kwargs', type=str, dest='mode_kwargs', action=DictAction, default={}, nargs='+')
    parser.add_argument('-t', '--test', dest='test', action='store_true', default=False)

    options = parser.parse_args()

    main(options)
