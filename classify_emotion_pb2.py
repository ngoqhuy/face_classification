# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: classify_emotion.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16\x63lassify_emotion.proto\"\"\n\x0cRequestImage\x12\x12\n\norig_image\x18\x01 \x01(\x0c\"%\n\x0fRequestEmotions\x12\x12\n\norig_image\x18\x01 \x01(\x0c\"\x1e\n\rRespondImages\x12\r\n\x05image\x18\x01 \x01(\t\"#\n\x0fRespondEmotions\x12\x10\n\x08\x65motions\x18\x01 \x01(\t2\x80\x01\n\rAsillaService\x12\x35\n\x12preporcess_emotion\x12\r.RequestImage\x1a\x0e.RespondImages\"\x00\x12\x38\n\x10\x63lassify_emotion\x12\x10.RequestEmotions\x1a\x10.RespondEmotions\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'classify_emotion_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REQUESTIMAGE._serialized_start=26
  _REQUESTIMAGE._serialized_end=60
  _REQUESTEMOTIONS._serialized_start=62
  _REQUESTEMOTIONS._serialized_end=99
  _RESPONDIMAGES._serialized_start=101
  _RESPONDIMAGES._serialized_end=131
  _RESPONDEMOTIONS._serialized_start=133
  _RESPONDEMOTIONS._serialized_end=168
  _ASILLASERVICE._serialized_start=171
  _ASILLASERVICE._serialized_end=299
# @@protoc_insertion_point(module_scope)