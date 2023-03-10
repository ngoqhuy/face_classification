# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import classify_emotion_pb2 as classify__emotion__pb2


class AsillaServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.preporcess_emotion = channel.unary_unary(
                '/AsillaService/preporcess_emotion',
                request_serializer=classify__emotion__pb2.RequestImage.SerializeToString,
                response_deserializer=classify__emotion__pb2.RespondImages.FromString,
                )
        self.classify_emotion = channel.unary_unary(
                '/AsillaService/classify_emotion',
                request_serializer=classify__emotion__pb2.RequestEmotions.SerializeToString,
                response_deserializer=classify__emotion__pb2.RespondEmotions.FromString,
                )


class AsillaServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def preporcess_emotion(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def classify_emotion(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AsillaServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'preporcess_emotion': grpc.unary_unary_rpc_method_handler(
                    servicer.preporcess_emotion,
                    request_deserializer=classify__emotion__pb2.RequestImage.FromString,
                    response_serializer=classify__emotion__pb2.RespondImages.SerializeToString,
            ),
            'classify_emotion': grpc.unary_unary_rpc_method_handler(
                    servicer.classify_emotion,
                    request_deserializer=classify__emotion__pb2.RequestEmotions.FromString,
                    response_serializer=classify__emotion__pb2.RespondEmotions.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'AsillaService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class AsillaService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def preporcess_emotion(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AsillaService/preporcess_emotion',
            classify__emotion__pb2.RequestImage.SerializeToString,
            classify__emotion__pb2.RespondImages.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def classify_emotion(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AsillaService/classify_emotion',
            classify__emotion__pb2.RequestEmotions.SerializeToString,
            classify__emotion__pb2.RespondEmotions.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
