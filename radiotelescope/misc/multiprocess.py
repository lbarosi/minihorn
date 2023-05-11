# -*- coding: utf-8 -*-
"""Fornece funções para rodar trabalhos paralelos e contgrolar sua execução.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 09.04.2022
"""
import time
import multiprocessing
import os
from time import perf_counter

def run_progress(target=None, interval=60, **kwargs):
    """Roda função thread em paralelo, enviando parâmetros kwargs e monitorando execução no intervalo de tempo especificado.

    Esta função não tem timeout.

    Args:
        thread (type): função a ser executada.
        interval (type): tempo para reportar andamento. Defaults to 60.
        **kwargs (type): passa parâmetros para `thread`.
    """
    process = multiprocessing.Process(target = target, kwargs = kwargs)
    process.start()
    pid = process.pid
    start = perf_counter()
    while process.is_alive():
        time.sleep(interval)
        print("processo {} rodando.".format(pid))
    end = perf_counter()
    elapsed = end - start
    print("processo terminado em {:2f} segundos.".format(elapsed))
    return


def run_detached(target=None, monitor=True, interval=None, **kwargs):
    """Roda em background.

    Roda o controle de execução do thread e o thread, cada um em um processo, deixando o sistema livre para continuar execução.

    Args:
        thread (type): função ou método a ser executado. Defaults to None.
        monitor (bool): Se `True` imprime monitoramento de execução em tempo especificado em kwargs. Defaults to True.
        *args (type): parâmetros para thread `*args`.
        **kwargs (type): parâmetros para thread `**kwargs`.
    """
    if monitor:
        process = multiprocessing.Process(
            target=run_progress,
            kwargs={"target":target, "interval":interval, **kwargs}
            )
        process.start()
    else:
        process = multiprocessing.Process(target=target, kwargs=kwargs, daemon=True)
        process.start()
    return


def run_daemon(thread=None, *args, **kwargs) -> multiprocessing.Process:
    """Roda o comando indicado em modo background, liberando a execução do resto do programa.

    Retorna o processo em execução.

    """
    try:
        process = multiprocessing.Process(
                                          target=thread,
                                          args=args,
                                          kwargs=kwargs,
                                          daemon=True
                                          )
        process.start()
    except OSError as error:
        logger.error("Detached program {} failed to execute: {}".format(command, error))
    return process
