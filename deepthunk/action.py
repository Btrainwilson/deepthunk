from typing import Dict, List, Any, Optional, Callable, Union
from tabulate import tabulate
import json

from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)


def fg256(n: int) -> str:
    return f"\033[38;5;{n}m"

RESET = "\033[0m"

class ActionSpace:
    def __init__(self):

        self.action_types = ["START", "STOP"]

        # vocab meta
        self.types: Dict[str, List[Any]] = {}
        self.actions: Dict[str, Dict[str, str]] = {}  # action -> param_name -> type
        self.handlers: Dict[str, Callable] = {}


    @property
    def num_actions(self):
        return len(self.action_types)

    def add_types(self, types: Dict[str, List[Any]]):
        self.types.update(types)

    def add_action(self, action_name: str, param_defs: Dict[str, str], handler: Optional[Callable] = None):

        if action_name in self.action_types:
            raise ValueError(f"Action '{action_name}' already exists.")

        self.action_types.append(action_name)

        self.actions[action_name] = param_defs
        self.handlers[action_name] = handler

    def compile(self):

        self.tokens = self.action_types
        tidx = len(self.tokens)
        self.tname_to_tidx = {a_name : i for i, a_name in enumerate(self.action_types)}
        self.tidx_to_tname = list(self.tname_to_tidx.keys())

        for type_name in self.types.keys():
            for i, type_val in enumerate(self.types[type_name]):
                tok = f"{type_name}_{i}"
                self.tname_to_tidx[tok] = tidx
                self.tidx_to_tname.append(tok)
                tidx += 1

    def encode(self, sequence: List[Dict[str, Any]]) -> List[int]:

        tokens = [self.tname_to_tidx['START']]
        for step in sequence:

            action = list(step.keys())[0]
            params = step[action]

            if not action in self.action_types:
                raise ValueError(f"Action {action} not supported.")

            tokens.append(self.tname_to_tidx[action])

            for param_name, param_val in params.items():
                ty = self.actions[action][param_name]
                i = self.types[ty].index(param_val)
                tok = f"{ty}_{i}"
                if not tok in self.tname_to_tidx:
                    raise ValueError(f"type_val {tok} not supported.")
                tokens.append(self.tname_to_tidx[tok])

        tokens.append(self.tname_to_tidx['STOP'])

        return tokens

    def decode(self, tokens: List[int]) -> List[Dict[str, Any]]:

        assert tokens[0] == self.tname_to_tidx['START']

        i = 1
        decoded = []
        while i < len(tokens):
            tid = tokens[i]
            if tid == self.tname_to_tidx['STOP']:
                break

            if tid < len(self.action_types):
                action = self.tidx_to_tname[tid]
            else:
                raise ValueError(f"Expected action token in front buffer, got id {tid}")

            param_defs = self.actions[action]
            i += 1
            params = {}

            for j, param_name in enumerate(param_defs):
                pid = tokens[i]
                if pid < len(self.action_types):
                    raise ValueError(f"Expected param token, got action token id {pid}")
                param_token = self.tidx_to_tname[pid]

                type_name, t_i = param_token.split('_')
                params[param_name] = self.types[type_name][int(t_i)]
                i += 1

            decoded.append({"type": action, "params": params})
        return decoded

    def run(self, sequence: List[Dict[str, Any]]):
        for step in sequence:
            action = step['type']
            params = step['params']
            if action not in self.handlers:
                raise ValueError(f"No handler for action '{action}'")
            self.handlers[action](**params)




    def print_vocab(self, verbose=False):
        print("\n" + fg256(217) + "=== Action Buffer (Actions and Parameters) ===" + RESET)
        action_rows = []
        for action, params in self.actions.items():
            param_str = ", ".join(
                f"{fg256(229)}{p}{RESET}:{fg256(79)}{t}{RESET}"
                for p, t in params.items()
            )
            action_rows.append([
                fg256(141) + str(self.tname_to_tidx[action]) + RESET,  # Mauve
                fg256(117) + action + RESET,                          # Sky
                param_str
            ])
        print(tabulate(action_rows, headers=[
            fg256(250) + "Token ID" + RESET,
            fg256(250) + "Action" + RESET,
            fg256(250) + "Parameters" + RESET
        ], tablefmt="fancy_grid"))

        print("\n" + fg256(217) + "=== Type Buffer (Parameter Value Tokens) ===" + RESET)
        type_rows = []
        for type_name, values in self.types.items():
            if verbose:
                for i, val in enumerate(values):
                    token = f"{type_name}_{i}"
                    token_id = self.tname_to_tidx.get(token, "N/A")
                    type_rows.append([
                        fg256(141) + str(token_id) + RESET,      # Mauve
                        fg256(117) + token + RESET,              # Sky
                        fg256(79) + type_name + RESET,           # Teal
                        str(val)
                    ])

            else:
                token = f"{type_name}_i"
                token_id_first = self.tname_to_tidx.get(f"{type_name}_{0}", "N/A")
                token_id_last = self.tname_to_tidx.get(f"{type_name}_{len(values) - 1}", "N/A")
                type_rows.append([
                    fg256(141) + str(token_id_first) + " - " + str(token_id_last) + RESET,      # Mauve
                    fg256(117) + token + RESET,              # Sky
                    fg256(79) + type_name + RESET,           # Teal
                    fg256(79) + "[" + str(values[0]) + " - " + str(values[-1]) + "]" + RESET,           # Teal
                ])
        print(tabulate(type_rows, headers=[
            fg256(250) + "Token ID" + RESET,
            fg256(250) + "Token" + RESET,
            fg256(250) + "Type" + RESET,
            fg256(250) + "Value" + RESET
        ], tablefmt="fancy_grid"))



    def print_run(self, sequence: List[Dict[str, Any]]):
        print("\n" + fg256(217) + "=== Executing Parameterized Actions ===" + RESET)

        for i, step in enumerate(sequence):
            action = step["type"]
            params = step["params"]

            action_str = fg256(117) + f"{action}" + RESET
            param_str = ", ".join(
                f"{fg256(229)}{k}{RESET}={fg256(79)}{v}{RESET}" for k, v in params.items()
            )

            print(f"{fg256(141)}[{i}]{RESET} {action_str}({param_str})")

    def save_json(self, path: str):
        data = {
            "action_types": self.action_types,
            "types": self.types,
            "actions": self.actions
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"{fg256(79)}Action space saved to {path}{RESET}")

    @classmethod
    def load_json(cls, path: str) -> "ActionSpace":
        with open(path, 'r') as f:
            data = json.load(f)

        instance = cls()
        instance.action_types = data["action_types"]
        instance.types = data["types"]
        instance.actions = data["actions"]
        instance.handlers = {k: None for k in instance.actions}  # Handlers must be re-bound manually
        instance.compile()
        print(f"{fg256(79)}Action space loaded from {path}{RESET}")
        return instance
