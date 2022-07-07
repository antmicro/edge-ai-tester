from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import RuntimeProtocol, ServerStatus
from kenning.runtimeprotocols.network import NetworkProtocol
from pathlib import Path
from runtimeprotocolbase import RuntimeProtocolTests
from typing import Tuple, List
import time
import json
import multiprocessing
import pytest
import random
import socket
import uuid


class TestNetworkProtocol(RuntimeProtocolTests):
    runtimeprotocolcls = NetworkProtocol
    host = ''
    port = 1235

    def generate_byte_data(self) -> Tuple[bytes, List[bytes]]:
        """
        Generates correct data for test.

        Returns
        -------
        Tuple[bytes, List[bytes]]:
            A tuple containing bytes stream and expected output
        """
        data = bytes()
        answer = list()
        for i in range(random.randint(1, 10)):
            times = random.randint(1, 10)
            answer.append(bytes())
            tmp_order = bytes()
            for j in range(times):
                number = (random.randint(1, 4294967295))
                num_bytes = number.to_bytes(4, byteorder='little',
                                            signed=False)
                tmp_order += num_bytes
                answer[i] += num_bytes
            length = len(tmp_order)
            length = length.to_bytes(4, byteorder='little', signed=False)
            data += length + tmp_order
        return data, answer

    def test_wait_for_activity(self, server, client):
        # Connect via Client
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.CLIENT_CONNECTED
        assert received_data is None

        # Send data
        data, _ = self.generate_byte_data()
        client.send_data(data)
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.DATA_READY
        assert received_data == [data]

        # Send error message
        client.send_message(MessageType.ERROR, b'')
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.DATA_READY
        assert received_data == [MessageType.ERROR.to_bytes()]

        # Send fully empty message
        class InvalidMessage:
            def to_bytes(self):
                return b''

        client.send_message(InvalidMessage(), b'')
        status, received_data = server.wait_for_activity()[0]
        assert received_data == [b'']
        assert status == ServerStatus.DATA_READY

        # Disconnect with client
        client.disconnect()
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert received_data is None

        # Timeout is reached
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.NOTHING and received_data is None

    def test_send_data(self, server, client):
        data, _ = self.generate_byte_data()
        assert client.send_data(data) is True

    def test_receive_data(self, server, client):
        data, _ = self.generate_byte_data()

        # Not initialized on server side
        with pytest.raises(AttributeError):
            server.receive_data(None, None)

        # Data is sent
        server.accept_client(server.serversocket, None)
        assert client.send_data(data) is True
        status, received_data = server.receive_data(None, None)
        assert status is ServerStatus.DATA_READY
        assert [data] == received_data

        # Client disconnected
        client.disconnect()
        status, received_data = server.receive_data(None, None)
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert received_data is None

    def test_accept_client(self):
        def connect():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            s.close()

        def run_test(protocol: RuntimeProtocol):
            output: ServerStatus
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(1)
                multiprocessing.Process(target=connect).start()
                output = protocol.accept_client(s, None)
                s.shutdown(socket.SHUT_RDWR)
            return output

        # Test there's already established connection
        protocol = self.initprotocol()
        protocol.socket = True
        assert run_test(protocol)[0] == ServerStatus.CLIENT_IGNORED

        # Test there was no connection yet
        protocol = self.initprotocol()
        assert run_test(protocol)[0] == ServerStatus.CLIENT_CONNECTED

        # Test attribute
        protocol = self.initprotocol()
        with pytest.raises(AttributeError):
            protocol.accept_client(None, None)

    def test_collect_messages(self):
        # valid data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        data, answer = self.generate_byte_data()
        status, output = protocol.collect_messages(data)
        assert output == answer and status == ServerStatus.DATA_READY

        # empty data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        status, output = protocol.collect_messages(b'')
        assert output is None and status == ServerStatus.NOTHING

        # wrong amount of bytes to be read
        data = (10*4).to_bytes(4, 'little', signed=False)
        data += (1).to_bytes(4, 'little', signed=False)
        status, output = protocol.collect_messages(data)
        assert output is None and status == ServerStatus.NOTHING

    def test_wait_send(self, server, client):
        server.accept_client(server.serversocket, None)
        data, answer = self.generate_byte_data()

        for i in range(10):
            # Send the data
            client_out = client.wait_send(data)
            assert client_out == len(data)

            # Recieve data
            server_status, server_data = server.receive_data(None, None)
            assert server_status == ServerStatus.DATA_READY
            assert server_data == answer

    def test_send_message(self, server, client):
        server.accept_client(server.serversocket, None)
        data, _ = self.generate_byte_data()

        assert client.send_message(MessageType.DATA, data=data) is True
        assert server.send_message(MessageType.DATA, data=data) is True
        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(MessageType.OK, data=b'')

    def test_receive_confirmation(self, server):

        def confirm(server: NetworkProtocol, return_list: list):
            """
            Waits for message and appends output to provided shared list.
            """
            output = server.receive_confirmation()
            return_list.append(output)

        manager = multiprocessing.Manager()
        shared_list = manager.list()
        client = self.initprotocol()
        client.initialize_client()
        server.accept_client(server.serversocket, None)

        cases = (((MessageType.OK, b''), (True, b'')),
                 ((MessageType.ERROR, b''), (False, None)),
                 ((MessageType.DATA, b''), (False, None)),
                 ((MessageType.MODEL, b''), (False, None)),
                 ((MessageType.PROCESS, b''), (False, None)),
                 ((MessageType.OUTPUT, b''), (False, None)),
                 ((MessageType.STATS, b''), (False, None)),
                 ((MessageType.QUANTIZATION, b''), (False, None)),
                 )

        # Check for every presented MessageType
        for message, expected in cases:
            thread = multiprocessing.Process(target=confirm,
                                             args=(server, shared_list))
            thread.start()
            client.send_message(*message)
            thread.join()
            assert shared_list[-1] == expected

        # Check if client is disconnected
        client.disconnect()
        output = server.receive_confirmation()
        assert output == (False, None)

    def test_upload_input(self, server, client):

        def upload(client, data, shared_list):
            output = client.upload_input(data)
            shared_list.append(output)

        server.accept_client(server.serversocket, None)
        data, _ = self.generate_byte_data()
        shared_list = (multiprocessing.Manager()).list()
        thread = multiprocessing.Process(target=upload,
                                         args=(client, data, shared_list))
        thread.start()

        # We have to wait somehow untill data is delivered
        # TODO: get rid of sleep
        time.sleep(1)
        status, received_data = server.receive_data(None, None)
        server.send_message(MessageType.OK, b'')
        assert status == ServerStatus.DATA_READY
        assert received_data == [(MessageType.DATA).to_bytes() + data]
        assert shared_list[0] is True

    def test_upload_model(self, server, client, tmpfolder):
        path = tmpfolder / uuid.uuid4().hex
        data, _ = self.generate_byte_data()

        def write_model(path: Path, data: bytes):
            """
            Writes bytes to file
            """
            with open(path, "wb") as file:
                file.write(data)
            return

        def receive_model(server: RuntimeProtocol, shared_list: list):
            status, received_model = server.receive_data(None, None)
            shared_list.append((status, received_model))
            output = server.send_message(MessageType.OK, b'')
            shared_list.append(output)

        write_model(path, data)
        shared_list = (multiprocessing.Manager()).list()
        thread_receive = multiprocessing.Process(target=receive_model,
                                                 args=(server, shared_list))
        server.accept_client(server.serversocket, None)
        thread_receive.start()
        assert client.upload_model(path) is True
        thread_receive.join()
        answer = [(MessageType.MODEL).to_bytes() + data]
        assert shared_list[0] == (ServerStatus.DATA_READY, answer)
        assert shared_list[1] is True

    def test_upload_quantization_details(self, tmpfolder, server, client):
        quantization_details = {1: 'one', 2: 'two', 3: 'three'}
        path = tmpfolder / uuid.uuid4().hex
        with open(path, 'w') as file:
            json.dump(quantization_details, file)

        def receive_quant(server: RuntimeProtocol, shared_list: list):
            status, received_model = server.receive_data(None, None)
            shared_list.append((status, received_model))
            output = server.send_message(MessageType.OK, b'')
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (server, shared_list)
        thread_receive = multiprocessing.Process(target=receive_quant,
                                                 args=args)
        server.accept_client(server.serversocket, None)
        thread_receive.start()
        assert client.upload_quantization_details(path) is True
        thread_receive.join()

        status, received_data = shared_list[0]
        message, json_file = received_data[0][:2], received_data[0][2:]
        assert message == (MessageType.QUANTIZATION).to_bytes()
        assert json_file.decode() == json.dumps(quantization_details)
        assert shared_list[1] is True

    def test_download_output(self, server, client):
        data, _ = self.generate_byte_data()
        server.accept_client(server.serversocket, None)

        def send_stats(server, data, shared_list):
            server.send_message(MessageType.OK, data)

        shared_list = (multiprocessing.Manager()).list()
        args = (server, data, shared_list)
        thread_send = multiprocessing.Process(target=send_stats, args=args)
        thread_send.start()
        status, downloaded_data = client.download_output()
        assert status is True
        assert downloaded_data == data

    def test_download_statistics(self, server, client):
        data = {'1': 'one', '2': 'two', '3': 'three'}
        server.accept_client(server.serversocket, None)

        def send_stats(server, data: dict, shared_list):
            output = server.download_statistics()
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (client, data, shared_list)
        thread_send = multiprocessing.Process(target=send_stats, args=args)
        thread_send.start()
        # TODO: Get rid of sleep
        time.sleep(1)
        status, message = server.receive_data(None, None)
        message_type, message = server.parse_message(message[0])
        if message_type == MessageType.STATS and message == b'':
            to_send = json.dumps(data).encode()
            output = server.send_message(MessageType.OK, to_send)
            assert output is True
        thread_send.join()
        assert shared_list[0].data == data


@pytest.mark.fast
class TestMessageType:
    def test_to_bytes(self):
        byte_num = (1).to_bytes(2, 'little', signed=False)
        assert MessageType.ERROR.to_bytes('little') == byte_num

    def test_from_bytes(self):
        byte_num = (1).to_bytes(2, 'little', signed=False)
        assert MessageType.ERROR == MessageType.from_bytes(byte_num, 'little')
