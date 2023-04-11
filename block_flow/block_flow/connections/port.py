from typing import TYPE_CHECKING
from typing import Type

if TYPE_CHECKING:
    from block_flow.blocks.block import Block


class PortData:
    def __init__(self, value):
        self._value = None

    @property
    def data(self):
        return self._value

    @data.setter
    def data(self, value):
        self._value = value


class Port:
    def __init__(self, block: "Block", data_type: Type, name: str = None):

        # Block to which the signal belongs
        self.block = block

        # Data Type
        self.data_type = data_type

        # Port name
        self.name = name

        # Port ID
        self.port_id = None

        # Signal value
        self._port_data = PortData(value=data_type())

        # Connected?
        self.connected = False


class InputPort(Port):
    def __init__(self, block: "Block", data_type: Type, name: str = None):
        super().__init__(block, data_type, name)

    @property
    def data(self):
        return self._port_data._value

    @data.setter
    def data(self, value):
        self._port_data._value = value


class OutputPort(Port):
    def __init__(self, block: "Block", data_type: Type, name: str = None):
        super().__init__(block, data_type, name)

        # List of connected signals
        self.connections = []

    def _connect(self, dest) -> bool:
        dest._port_data = self._port_data

        return True

    @property
    def data(self):
        return self._port_data._value

    @data.setter
    def data(self, value):
        self._port_data._value = value
