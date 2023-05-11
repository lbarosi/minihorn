# -*- coding: utf-8 -*-
""" Módulo com miscelânea de funções para informações de rede e execução de comandos de sistema.

AUTHOR: Luciano Barosi
DATE: 09.04.2022

"""
#------------------
# Stdlib imports
import getpass
import os
import logging
import platform
import re
import shlex
import socket
import subprocess
import uuid
#------------------
# third party imports
import ifaddr
import ifcfg
import io
import ipaddress
import lxml.etree
import pandas as pd
import paramiko
from paramiko import SSHClient, AutoAddPolicy
from paramiko.pkey import PKey
from paramiko.auth_handler import AuthenticationException
from paramiko.auth_handler import SSHException
import pexpect
#from scp import SCPClient, SCPException
import xmltodict
# Module functions
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
#------------------
# local imports
#------------------
class RemoteClient:
    """Extensão de cliente remoto baseado em paramiko."""

    def __init__(self, host=None, port=22, user=None, password=None, remote_folder=None):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.client = None
        self.remote_folder = remote_folder
        return

    def connect(self):
        """conexão ssh."""
        if not self.user:
            self.user = input("Nome do Usuário: ")
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.host, username=self.user)
            self.client = client
        except AuthenticationException as err:
            if not self.password:
                print("digite senha para usuário {}:".format(self.user))
                self.password = getpass.getpass()
            client.connect(self.host, username=self.user, password=self.password)
            self.client = client
        except SSHException as error:
            print("error")
            return
        return self

    def _deploy_key(self, key=None):
        """ Instala chaves locais para ssh."""
        command_key = "echo " + str(key) + " > ~/.ssh/authorized_keys"
        _, _, _ = self.client.exec_command('mkdir -p ~/.ssh/')
        self.client.exec_command(command_key)
        self.client.exec_command('chmod 644 ~/.ssh/authorized_keys')
        self.client.exec_command('chmod 700 ~/.ssh/')
        return None

    def deploy_local_key(self, key=None):
        """Prepara conexão, verifica e instala chaves ssh."""
        if not isinstance(self.client, paramiko.SSHClient):
            self.connect()
        else:
            if not key:
                key = str(open(os.path.expanduser("~/.ssh/id_rsa.pub")).read()).strip()
            try:
                self._deploy_key(key=key)
            except SSHException as error:
                print(error)
        return self

    def run(self, command=None, stdout=True):
        """Roda comando remoto."""
        if not isinstance(self.client, paramiko.SSHClient):
            self.connect()
        else:
            if stdout:
                stdin, stdout, stderr = self.client.exec_command(command)
                if not stdout.channel.recv_exit_status():
                    print("Comando executado com sucesso.")
                else:
                    print("Execução do comando falhou.")
                response = stdout.readlines()
                for line in response:
                    print(f"INPUT: {command} | OUTPUT: {line}")
            else:
                self.client.exec_command(command)
                response = None
        return response

def run_command(command: str) -> tuple[str, str]:
    """Executa comando shell passado como argumento utilizando biblioteca `subprocess`.

    Função reporta erro no log.

    Args:
        command (str): string de comando escrita da mesma forma que se escreveria em linha de comando.

    Returns:
        tuple[str, str]: saída padrão e código de erro informado para o comando executado.
    """
    process = subprocess.Popen(
                                shlex.split(command),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                                )
    out, err = process.communicate()
    if err:
        err = err.decode("utf-8")
        logger.error("Error running command {}:{}".format(command, err))
    return (out, err)

def get_OS():
    """Obtem nome do sistema operacional em uso na máquina local.

    Returns:
        str: Nome do sistema operacional local.

    """
    result = platform.system()
    return result

def get_remote_OS(IP=None, port=22, user=None):
    """Obtem nome do sistema operacional de máquina remota utilizando conexão por SSH.

    Args:
        IP (str): Endereço da máquina remota `IP`. Defaults to None.
        port (int): `port` utilizada para SSH. Defaults to 22.
        user (str): nome do `user` para SSH. Defaults to None.

    Returns:
        str: Nome do sistema operacional.

    """
    command_os = "python -c 'import platform; print(platform.system())'"
    saida = RemoteClient(
        host=IP, user=user, port=port
        ).deploy_local_key().run(command_os)
    result = "".join(saida).strip()
    return result

def _iface_type(iface_name):
    """Determina o tipo de interface de rede baseada no formato systemd de endereços previsíveis.

    tipos disponíveis:
    - loopback
    - WAN
    - LAN
    - virtual
    - docker
    - None
    """
    if iface_name == "lo":
        result = "loopback"
    elif iface_name[0] == "v":
        result = "virtual"
    elif "docker" in iface_name:
        result = "docker"
    elif iface_name[0] == "e":
        result = "LAN"
    elif iface_name[0] == "w":
        result = "WAN"
    else:
        result = None
    return result

def get_iface(private=True, interfaces=["WAN"]):
    """Obtem todas as interfaces de rede e respectivos IPs, classificando-os em privados ou públicos.

    Args:
        private (bool): seleciona endereços privados (`True`), públicos (`False`) ou todos (qualquer outra opção). Defaults to True.
        interfaces (type): **lista** de tipos de interface `["WAN", "LAN", "loopback", "virtual", "docker", None]``. Defaults to ["WAN"].

    Returns:
        pd.DataFrame:
    """
    if private == True:
        private = [True]
    if private == False:
        private = [False]
    else:
        private = [True, False]
    adapters = ifaddr.get_adapters()
    IPs = [ ips.ip for adapter in adapters for ips in adapter.ips]
    names = [ips.nice_name for adapter in adapters for ips in adapter.ips ]
    df_IP = pd.DataFrame({"interfaces":names, "IP":IPs})
    df_IP = df_IP[ [isinstance(IP, str) for IP in df_IP.IP]]
    df_IP["type"] = df_IP.interfaces.apply(_iface_type)
    df_IP["private"] = df_IP.IP.apply(lambda row: ipaddress.ip_address(row).is_private)
    types = [None]
    df = df_IP.loc[df_IP.type.isin(interfaces) & df_IP.private.isin(private)]
    return df

def get_ip(private=True, interface=None):
    """Obtem o IP da máquina local conectado em uma interface definida no argumento.

    Args:
        private (bool): seleciona endereços privados (`True`), públicos (`False`) ou todos (qualquer outra opção). Defaults to True.
        interface (type): seleciona tipo de interface `["WAN", "LAN", "loopback", "virtual", "docker", None]``. Defaults to ["WAN"].

    Returns:
        str: IP deve ser strin ou retorna None.

    """
    df = get_iface(private, interfaces=[interface])
    result = df[df.type==interface].IP.values[0]
    if isinstance(result, str):
        return result
    else:
        return

def get_MAC():
    """Endereço MAC da máquina local.

    Returns:
        str: Endereço MAC.

    """
    result = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
    return result

def get_remote_MAC(IP=None):
    """Endereço MAC de máquina remota obtido com nmap.

    Args:
        IP (str): endereço da máquina remota `IP`. Defaults to None.

    Returns:
        str: Endereço MAC.

    """
    # Comando NMAP com saída em XML para realizar parsing da informação.
    # Este nmap na máquina local NÃO retorna o endereço MAC.
    command = "nmap --privileged -sS " + str(IP) + " -oX -"
    saida, err = run_command(command)
    # Parsing de XML precisa cuidar da codificação, usando biblioteca lxml e codoficação utf-8
    root = lxml.etree.parse(io.BytesIO(saida))
    result = (root.find("host").findall("address")[1].attrib)["addr"]
    return result

def get_netmask(IP = None):
    """Determina máscara de rede associada a endereço IP.

    Args:
        IP (str): `IP`. Defaults to None.

    Returns:
        str: Máscara de rede.

    """
    for name, interface in ifcfg.interfaces().items():
        result = "255.255.255.0"
        try:
            if interface['inet4'][0] == IP:
                result = str(interface["netmask"])
        except IndexError as err:
            pass
    return result

def get_subnet(IP = None):
    """Determina LAN do IP indicado.

    Args:
        IP (str): `IP`. Defaults to None.

    Returns:
        str: subrede no formato CIDR.

    """
    netmask = get_netmask(IP)
    net = ipaddress.ip_interface(str(IP) + "/" + netmask)
    result = str(net)
    return result

def get_raspberry(IP=None):
    """Cria dataframe com informação de todos os dispositivos raspberry encontrados na rede local."""
    local_IP = IP
    subnet = get_subnet(IP = local_IP)
    filename = "./net_scan.xml"
    scan_network = run_command("nmap --privileged -sn " + subnet + " -oX " + filename);
    tree = lxml.etree.parse(filename)
    try:
        os.remove(filename)
    except OSError as e:
        pass
    hosts = pd.DataFrame([{"host":ii, **host.attrib} for ii, child in enumerate(tree.findall("host")) for host in child.iterdescendants() if host.tag == "address" ])
    try:
        raspberry = hosts[hosts.host.isin(hosts[hosts.vendor.str.contains("PI", case = False, na=False)].host.tolist())]
    except AttributeError as err:
        raspberry = pd.DataFrame()
    result = raspberry
    return result

def report_raspberry(raspberry):
    """Retorna lista de IPs de dispositos raspberry encontrados em dataframe."""
    if not raspberry.empty:
        print("Encontrados {} dispositivos raspberry.".format(raspberry.host.nunique()))
        for group, df in raspberry.groupby("host"):
            for ii, row in df.iterrows():
                if row["addrtype"] == "ipv4":
                    print("Rasberry {} \n".format(group))
                    print("Endereço IP: {}".format(row.addr))
                if row["addrtype"] == "mac":
                    print("Endereço MAC: {}".format(row.addr))
        result = raspberry[raspberry.addrtype == "ipv4"].addr.unique()
    else:
        print("Nenhum Raspberry Encontrado")
        result = []
    return result
