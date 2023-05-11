# -*- coding: utf-8 -*-
"""Fornece funções para controle de controladores de backends, usualmente computadores ou raspberries que recebem um determinado conjunto de comandos e implementam métodos para entrada e saída de dados e para início, parada e configuração do receptor backend ligado ao controller.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 09.04.2022
"""
#------------------
# Stdlib imports
from abc import ABC, abstractmethod
import fcntl
import io
import logging
import os
import re
import sys
import uuid
#------------------
# third party imports
import lxml
import pandas as pd
#------------------
# local imports
#------------------
import radiotelescope.netutils.netutils as netutils
#------------------
logger = logging.getLogger(__name__)
# Equivalent of the _IO('U', 20) constant in the linux kernel.
USBDEVFS_RESET = ord('U') << (4*2) | 20

class Controller(ABC):
    """Classe para controlador, uma abstração de computador com IS arbitrário, usuário e conexão de rede.

    Args:
        IP (str): endereço `IP`. Defaults to None.
        MAC (str): endereço `MAC`. Defaults to None.
        OS (str): sistema operacional `OS`. Defaults to None.
        name (tstr): nome `name`. Defaults to None.
        user (str): usuário local `user`. Defaults to None.
        folder (atr): pasta para armazenamento de arquivos `folder`. Defaults to None.
        remote (Controller): outra instância de controlador, na mesma rede `remote`. Defaults to None.
        connected (Bool): indicador de conectividade entre host e remoto `connected`. Defaults to False.

    Attributes:
        MAC
        OS
        folder
        remote
        connected
    """

    def __init__(self, interface=None, IP=None, MAC=None, OS=None, name=None,
                 user=None, local_folder = None, remote=None, remote_folder=None, remote_IP=False, remote_port=22):
        """Método init da classe. Verifique os parâmetros da classe."""
        self._IP = IP
        self._name = name
        self._user = user
        self._remote = remote
        self._local_folder = local_folder
        self.interface = interface
        self.MAC = MAC
        self.OS = OS
        self.remote = remote
        self.remote_folder = remote_folder
        self.remote_IP = remote_IP
        self.remote_port = remote_port
        return None

    @property
    def name(self):
        """Propriedade: nome da instância."""
        return self._name

    @name.setter
    def name(self, name):
        """Propriedade: nome da instância."""
        self._name = name

    @property
    def user(self):
        """Propriedade: usuário local."""
        return self._user

    @user.setter
    def IP(self, user):
        """Propriedade: usuário local."""
        self._user = user

    @property
    def IP(self):
        """Propriedade: endereço IP."""
        return self._IP

    @IP.setter
    def IP(self, IP):
        """Propriedade: endereço IP."""
        self._IP = IP

    @property
    def remote(self):
        """Propriedade: instância Controller remoto."""
        return self._remote

    @remote.setter
    def remote(self, remote):
        """Propriedade: instância Controller remoto."""
        self._remote = remote

    @property
    def local_folder(self):
        """Propriedade: nome da instância."""
        return self._local_folder

    @local_folder.setter
    def local_folder(self, local_folder):
        """Propriedade: nome da instância."""
        self._local_folder = local_folder


    @abstractmethod
    def connect(self, IP=None, user=None):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def scan_LAN(self):
        pass

    @abstractmethod
    def run(self, command = None):
        pass

    @abstractmethod
    def run_remote(self, IP=None, user=None, command=None):
        pass

    @abstractmethod
    def sync_files(self, origin=None):
        pass

class LinuxBox(Controller):
    """Classe para controlador com sistema operacional LinuxLike.

    Args:
        IP (str): endereço `IP`. Defaults to None.
        MAC (str): endereço `MAC`. Defaults to None.
        OS (str): sistema operacional `OS`. Defaults to None.
        name (tstr): nome `name`. Defaults to None.
        user (str): usuário local `user`. Defaults to None.
        folder (atr): pasta para armazenamento de arquivos `folder`. Defaults to None.
        remote (Controller): outra instância de controlador, na mesma rede `remote`. Defaults to None.

    Attributes:
        MAC
        OS
        folder
        remote
    """

    def __init__(self, **kwargs):
        """Init da classe LinuxBox. Verifique as propriedades e atributos na documentação da classe."""
        super().__init__(**kwargs)

    def get_info(self):
        """Mostra informações do controlador. É necessário especificar `IP` ou `interface`."""
        print("Verificando dados da máquina local:")
        if not self.IP:
            if self.interface:
                self.IP = netutils.get_ip(interface=self.interface)
            else:
                print("É preciso especificar interface ou IP.")
        self.MAC = netutils.get_MAC()
        self.OS = netutils.get_OS()

        print("IP: {}".format(self.IP))
        print("MAC: {}".format(self.MAC))
        print("OS: {}".format(self.OS))
        return self

    def connect(self, IP=None, key=None):
        """Estabelece conexão entre o controlador e cliente SSH remoto.

        Args:
            IP (str): `IP`. Defaults to None.
            key (str): endereço de arquivo com chave openssl pública para instalar no cliente remoto.. Defaults to None.
        """
        if isinstance(self.remote, netutils.RemoteClient):
            print("Remoto já conectado em IP {}".format(self.remote_IP))
        else:
            if not IP:
                IP = self.remote_IP
            self.remote = netutils.RemoteClient(host=IP, port=self.remote_port, user=self.user, remote_folder=self.remote_folder).connect()
            self.remote.deploy_local_key(key=key)
        return self

    def run(self, command = None):
        """Executa instruções na máquina local."""
        result, err = netutils.run_command(command=command)
        return result

    def run_remote(self, command=None, stdout=True):
        """Executa comandos na máquina remota via ssh."""
        try:
            remote = self.remote
            result = remote.run(command=command, stdout=stdout)
        except AttributeError:
            print("Remoto não definido.")
            return
        return result

    def scan_LAN(self, IP=None):
        """Determina rede local."""
        if not IP:
            IP = self.IP
        if IP:
            subnet = netutils.get_subnet(IP = IP)
            command = "nmap --privileged -sn " + subnet + " -oX -"
            saida, err = netutils.run_command(command)
            # Parsing de XML precisa cuidar da codificação, usando biblioteca lxml e codoficação utf-8
            root = lxml.etree.parse(io.BytesIO(saida))
            hosts = pd.DataFrame(
                [ {"host":ii, **host.attrib}
                    for ii, child in enumerate(root.findall("host"))
                    for host in child.iterdescendants()
                    if host.tag == "address"
                    ]
                )
            result = pd.merge(
                hosts[hosts.addrtype == "ipv4"].rename(columns = {"addr":"IP"})[["host", "IP"]],
                hosts[hosts.addrtype == "mac"].rename(columns = {"addr":"MAC"})[["host", "MAC"]]
                ).drop(columns = "host")
        else:
            print("Endereço IP não definido.")
            result = None
        return result

    def sync_files(self, origin="remote"):
        """Sincroniza pastas no cliente e no servidor.

        Args:
            origin (str): `local` ou `remote` indica a direção de sincronização. Defaults to "remote".
        """
        if origin == "remote":
            print("Sincronizando dados: {} -> {}.".format(self.remote_folder, self.local_folder))
            command = "rsync -a " + str(self.user) + "@" + str(self.remote_IP) + ":" + str(self.remote_folder) + " " + str(self.local_folder)
            result = self.run(command=command)
        elif origin == "local":
            print("Sincronizando dados: {} -> {}.".format(self.local_folder, self.remote_folder))
            command = "rsync -a " + str(self.local_folder) + " " + str(self.user) + "@" + str(self.remote_IP) + ":" + str(self.remote_folder)
            result = self.run(command=command)
        else:
            print("Você deve escolher `origin` local ou remote.")
        return

    def get_device(self, DEVICE):
        out = self.run("lsusb")
        lines = out.decode('utf-8').splitlines()
        result = None
        for line in lines:
            if DEVICE in line:
                parts = line.split()
                bus = parts[1]
                dev = parts[3][:3]
                result = '/dev/bus/usb/%s/%s' % (bus, dev)
        return result

    def reset_device(self, DEVICE):
        dev_path = self.get_device(DEVICE)
        try:
            fd = os.open(dev_path, os.O_WRONLY)
            fcntl.ioctl(fd, USBDEVFS_RESET, 0)
            os.close(fd)
            return True
        except TypeError:
            logger.error("Dispositivo não encontrado")
            return False
