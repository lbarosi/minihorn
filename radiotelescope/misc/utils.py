# -*- coding: utf-8 -*-
"""
Este módulo contém miscelânea de funções úteis para o pacote radiotelescope.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 09.04.2022
"""
import astropy.units as u

from astropy.units.quantity import Quantity as quantity
from datetime import timedelta
import io
import pandas as pd
import radiotelescope.netutils as netutils

def parse_time(time_object):
    """Retorna duração em segundos a partir de algum objeto temporal."""
    if isinstance(time_object, quantity):
        duration = time_object.to(u.s).value
    elif isinstance(time_object, timedelta):
        duration = time_object.total_seconds()
    elif isinstance(time_object, str):
        duration = pd.to_timedelta(time_object).value/1e9
    else:
        print("valor informado não é tipo tempo.")
    return duration


def get_PID(command):
    """Determina o número dos processo rodando na máquina local, emulando o comportamento de `ps aux |grep `.

    Returns:
        list: lista contendo números dos procesos ou lista vazia se nenhum processo encontrado.
    """
    probe_command = "ps -Ao pid,user,comm"
    process_name = command
    response, _ = netutils.run_command(probe_command)
    # very roundabout way to get all hits in ps command into a dataframe for proper parsing.
    df_ps = pd.read_table(io.StringIO(response.decode("utf-8")),
                          header=None)[0].str.split(' +', expand=True)
    PID = df_ps[df_ps[3].str.contains(process_name)][1].values.tolist()
    return PID

def is_running(command, interval=5, timeout = 300, monitor=True):
    """Determina se `command` esta sendo executado na máquina.

    Args:
        command (str): nome do processo para `ps`.
        interval (float): `interval` em segundos para reportas monitoramento. Defaults to 5.
        timeout (float): `timeout` em segundos para abortar. Defaults to 300.
        monitor (bool): `monitor` imprime monitoramento periódico na stdout. Defaults to True.
    """
    start = time.perf_counter()
    while True:
        response = utils.get_PID(command)
        if not response:
            print("Processo não está mais rodando")
            break
        time.sleep(interval)
        duration = time.perf_counter() - start
        if monitor:
            print("processo {} rodando. t = {:.2f}".format(command, duration))
        if duration > timeout:
            print("Timeout")
            break
    return
