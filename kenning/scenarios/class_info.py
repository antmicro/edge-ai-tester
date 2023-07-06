#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script that provides information about a given kenning class.

More precisely, it displays:
    - module and class docstring
    - imported dependencies, including information if they are available or not
    - supported input and output formats (lots of the classes provide such
     information one way or the other)
    - node's parameters, with their help and default values
"""
import argparse
import ast
import importlib
import os.path
import sys
from typing import Union, List

import astunparse
from isort import place_module

KEYWORDS = ['inputtypes', 'outputtypes', 'arguments_structure']


class _Argument:
    """
    Class representing an argument. Fields that are empty are not displayed.
    """

    def __init__(self):
        self.name = ''
        self.argparse_name = ''
        self.description = ''
        self.required = ''
        self.default = ''
        self.nullable = ''
        self.type = ''
        self.enum: List[str] = []

    def __repr__(self):
        lines = [f'* {self.name}']

        if self.argparse_name:
            lines.append(f'  * argparse name: {self.argparse_name}')
        if self.type:
            lines.append(f'  * type: {self.type}')
        if self.description:
            lines.append(f'  * description: {self.description}')
        if self.required:
            lines.append(f'  * required: {self.required}')
        if self.default:
            lines.append(f'  * default: {self.default}')
        if self.nullable:
            lines.append(f'  * nullable: {self.nullable}')

        if len(self.enum) != 0:
            lines.append('  * enum')
        for element in self.enum:
            lines.append(f'    * {element}')

        return '\n'.join(lines)


def print_class_module_docstrings(syntax_node: Union[ast.ClassDef, ast.
                                  Module]):
    """
    Displays docstrings of provided class or module

    Parameters
    ----------
    syntax_node: Union[ast.ClassDef, ast.Module]
        Syntax node representing a class or module
    """

    docstring = ast.get_docstring(syntax_node, clean=True)

    if not docstring:
        return

    docstring = '\n'.join(
        ['\t' + docstr for docstr in docstring.strip('\n').split('\n')])

    if isinstance(syntax_node, ast.ClassDef):
        print(f'Class: {syntax_node.name}\n')
        print(f'{docstring}')

    if isinstance(syntax_node, ast.Module):
        print('Module description:\n')
        print(docstring)

    print('')


def get_dependency(syntax_node: Union[ast.Import, ast.ImportFrom]) \
        -> str:
    """
    Extracts a dependency from an import syntax node and checks whether the
     dependency is satisfied. It also skips internal kenning modules

    Parameters
    ----------
    syntax_node: Union[ast.Import, ast.ImportFrom]
        An assignment like `from iree.compiler import version``
    """
    for dependency in syntax_node.names:
        module_path = ''
        dependency_path = ''
        if isinstance(syntax_node, ast.ImportFrom):
            dependency_path = f'{syntax_node.module}.{dependency.name}'
            module_path = f'{syntax_node.module}'

        if isinstance(syntax_node, ast.Import):
            dependency_path = f'{dependency.name}'
            module_path = dependency_path

        if module_path == '' or dependency_path == '':
            return ''

        try:
            importlib.import_module(module_path)

            if 'kenning' in dependency_path:
                return ''

            if place_module(module_path) == 'STDLIB':
                return ''  # TODO print standard modules with a script argument

            return '* ' + dependency_path
        except ImportError or ModuleNotFoundError as e:
            return f'* {dependency_path} - Not available (Reason: {e})'


def print_input_specification(syntax_node: ast.Assign):
    """
    Displays information about the input specification as bullet points

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `inputtypes = []`
    """

    if isinstance(syntax_node.value, ast.List) \
            and len(syntax_node.value.elts) == 0:
        return

    for input_format in syntax_node.value.keys:
        print(f'* {input_format.value}')


def print_output_specification(syntax_node: ast.Assign):
    """
    Displays information about the output specification as bullet points

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `outputtypes = ['iree']`
    """
    for output_format in syntax_node.value.elts:
        print(f'* {output_format.value}')


def print_arguments_structure(syntax_node: ast.Assign, source_path: str):
    """
    Displays information about the argument structure specification as
     bullet points

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `arguments_structure = {'compiler_args': {}}`
    """
    for argument, argument_specification_dict in zip(syntax_node.value.keys,
                                                     syntax_node.value.values):
        argument_object = _Argument()

        argument_object.name = argument.value

        for key, value in zip(argument_specification_dict.keys,
                              argument_specification_dict.values):

            if isinstance(value, ast.Call) and value.func.id == 'list':
                argument_list_variable = astunparse.unparse(value)\
                    .strip()\
                    .removeprefix("'")\
                    .removesuffix("'")\
                    .replace('list(', '')\
                    .replace('.keys())', '')

                argument_keys, argument_type = evaluate_argument_list_of_keys(
                    argument_list_variable,
                    source_path)

                argument_object.enum = argument_keys
                argument_object.type = argument_type

            elif key.value == "enum":
                argument_list_variable = astunparse\
                    .unparse(value)\
                    .strip()\
                    .removeprefix("'")\
                    .removesuffix("'")

                enum_list, argument_type = evaluate_argument_list(
                    argument_list_variable,
                    source_path)

                argument_object.enum = enum_list
                argument_object.type = argument_type

            else:

                key_str = astunparse.unparse(key)\
                    .strip()\
                    .removeprefix("'")\
                    .removesuffix("'")

                value_str = astunparse.unparse(value)\
                    .strip()\
                    .removeprefix("'")\
                    .removesuffix("'")

                argument_object.__setattr__(key_str, value_str)

        print(argument_object)


def evaluate_argument_list_of_keys(argument_list_name: str, source_path: str) \
        -> tuple[List[str], str]:
    with open(source_path) as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    argument_list_keys = []
    argument_type = ''

    for node in syntax_nodes:
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.targets[0], ast.Name):
            continue

        if not node.targets[0].id == argument_list_name:
            continue

        for key in node.value.keys:
            argument_list_keys.append(key.value)

        argument_type = f'List[{type(node.value.keys[0].value).__name__}]'

        break

    return argument_list_keys, argument_type


def evaluate_argument_list(argument_list_name: str, source_path: str) \
        -> tuple[List[str], str]:
    with open(source_path) as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    enum_elements = []
    argument_type = ''

    for node in syntax_nodes:
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.targets[0], ast.Name):
            continue

        if not node.targets[0].id == argument_list_name:
            continue

        for element in node.value.elts:
            enum_elements.append(element.value)

        argument_type = f'List[{type(node.value.elts[0].value).__name__}]'
        break

    return enum_elements, argument_type


def generate_class_info(target: str):
    """
    Wrapper function that handles displaying information about a class

    Parameters
    ----------
    target: str
        Target class path or module name e.g. either `kenning.core.flow` or
         `kenning/core/flow.py`
    """

    target_path = target
    if not target.endswith('.py'):
        target_path = target.replace('.', '/')
        target_path += '.py'

    if not os.path.exists(target_path):
        print('This class does not exist')
        return

    with open(target_path) as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    class_nodes = []
    dependency_nodes = []

    input_specification_node = None
    output_specification_node = None
    arguments_structure_node = None

    for node in syntax_nodes:
        if isinstance(node, (ast.ClassDef, ast.Module)):
            class_nodes.append(node)

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dependency_nodes.append(node)

        if isinstance(node, ast.Assign) and \
                isinstance(node.targets[0], ast.Name):
            if node.targets[0].id not in KEYWORDS:
                continue

            if node.targets[0].id == KEYWORDS[0]:
                input_specification_node = node
            if node.targets[0].id == KEYWORDS[1]:
                output_specification_node = node

            if node.targets[0].id == KEYWORDS[2]:
                arguments_structure_node = node

    for node in class_nodes:
        print_class_module_docstrings(node)

    print('Dependencies:')
    dependencies: List[str] = []
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    for node in dependency_nodes:
        dependency_str = get_dependency(node)
        if dependency_str == '':
            continue
        dependencies.append(dependency_str)

    [print(dep_str) for dep_str in list(set(dependencies))]

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    print('')

    print("Input formats:")
    if input_specification_node:
        print_input_specification(input_specification_node)
    print('')

    print("Output formats:")
    if output_specification_node:
        print_output_specification(output_specification_node)
    print('')

    print("Arguments specification:")
    if arguments_structure_node:
        print_arguments_structure(arguments_structure_node, target_path)


def main(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument(
        'target',
        help='',
        type=str
    )

    args = parser.parse_args(argv[1:])

    generate_class_info(target=args.target)


if __name__ == '__main__':
    main(sys.argv)
