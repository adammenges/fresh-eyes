from flask import Flask
from collections import defaultdict, OrderedDict
import time
from functools import (reduce, wraps)
import re
import os
import logging
from threading import RLock

# ML
import tensorflow as tf
from keras.backend import learning_phase
from keras.losses import get as get_keras_loss
from keras.metrics import categorical_accuracy
from keras.layers import Input
import numpy as np

def save(self, save_path, saver):
		print("Saving the graph...")
		save_start = time.time()
		with self.lock:
			saver.save(
				sess=self.session,
				save_path=save_path,
				global_step=self.global_step_variable,
				meta_graph_suffix='meta',
				write_meta_graph=True,
				write_state=True
			)
		print("Saving took {!s}".format(make_time_units_string(time.time() - save_start)))

def load_or_init_variables(self, save_dir, saver, table_feed_dict=None):
		"""
		Tries to instantiate our variables from the latest checkpoint, or from their init functions.
		Also initializes embedding tables.
		"""
		if self.cleanup:
			print("Attempting to init variables when cleanup flag set, returning instead.")
			return
		# not thread safe, so write lock around the function
		_init_start = time.time()
		if table_feed_dict is None:
			table_feed_dict = dict()

		with self.lock:
			with self.tf_graph.as_default():
				with self.tf_graph.device(device_name_or_function=self._device):
					# check if there's a smallest loss dir, if so, use that
					# smallest_loss_dir = os.path.join(save_dir, 'smallest_loss')
					# save_dir = smallest_loss_dir if os.path.isdir(smallest_loss_dir) else save_dir

					# attempt to load our vars from saver
					latest_checkpoint = None
					if saver is not None:
						try:
							print("trying to load saved variables...")
							latest_checkpoint = tf.train.latest_checkpoint(save_dir)  # TODO why changing gpu/cpu(?) kills this
							if latest_checkpoint is None:
								raise ValueError("no checkpoint found in {!s}".format(save_dir))
							try:
								saver.restore(self.session, save_path=latest_checkpoint)
								print("restored vars from {!s}".format(latest_checkpoint))
							except Exception as e:
								print("couldn't restore directly from checkpoint, trying optimistic restore...")
								restore_ops = self._optimistic_restore_ops(checkpoint=latest_checkpoint, var_list=saver._var_list)
								if len(restore_ops) > 0:
									self.session.run(restore_ops)
									print("restored {!s} vars from {!s}".format(len(restore_ops), latest_checkpoint))
								else:
									print("no vars matched from {!s}".format(latest_checkpoint))
						except Exception as e:
							print("couldn't reload from checkpoint: " + str(e))
					# initialize da vars
					var_list = self.tf_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
					# if loaded:
					# first, assign the global step (extract from checkpoint filename)
					if latest_checkpoint:
						step_val = int(os.path.basename(latest_checkpoint).split('-')[1])
						print("assigning global step to be {!s}".format(step_val))
						self.session.run(tf.assign(ref=self.global_step_variable, value=step_val))
					# only init those uninitialized vars if we loaded from saver
					name_to_var = {var.name.split(":")[0]: var for var in var_list}
					uninitialized_vars = tf.report_uninitialized_variables(var_list)
					uninitialized_var_names = self.session.run(uninitialized_vars)
					var_list = [name_to_var.get(str(var_name, 'utf-8')) for var_name in uninitialized_var_names]
					if len(var_list) > 0:
						print("initializing variables... {}".format([v.name for v in var_list]))
						init = tf.variables_initializer(var_list)
						self.session.run(init)
					# initialize tables
					print("initializing tables...")
					try:
						self.session.run(tf.tables_initializer(), feed_dict=table_feed_dict)
					except tf.errors.FailedPreconditionError:
						print("we already initialized the tables, silly :)")
					except Exception as e:
						print("table init error: {!s}".format(e))
		print("Variables load/init took {!s}".format(make_time_units_string(time.time() - _init_start)))

def _construct_feed_dict(self, x=None, y=None, seed=None, mode='train'):
		"""
		Creates the feed dictionary to pass through a sess.run() function
		"""
		if seed is None:
			seed = np.random.randint(100000)
		# print(f'_construct_feed_dict seed is {seed}')

		x = x or dict()
		y = y or dict()
		y = self._fix_ndims_y(y)
		data_items = x
		data_items.update(y)

		# construct our  feed dictionary from the x and y dictionaries (now data_items)
		# as well as setting our other placeholders
		feed_dict = dict()

		# go through the input and target placeholders, and map the tensor to the real value!
		# also, find the batch size and make sure all the batch sizes are the same!
		actual_batch_size = None

		for data_key, data_val in data_items.items():
			if data_val is None:
				raise Exception("data_val is none in _construct_feed_dict, usually this means something has failed when fetching from the db")
			feed_dict[self.tensors.get(data_key).value] = data_val
			batch_size = len(data_val)
			# check if the batch size is the same as all the other ones seen so far
			if actual_batch_size is None:
				actual_batch_size = batch_size
			# if they aren't the same, raise a value error!
			if batch_size != actual_batch_size:
				msg = "Found different data batch sizes! Was looking at data id {!s} and found batch size {!s} compared to others of {!s}".format(data_key, batch_size, actual_batch_size)
				print(msg)
				raise ValueError(msg)
		# other placeholders
		feed_dict[self.training_placeholder] = 1 if mode == 'train' else 0
		feed_dict[self.batch_size_placeholder] = actual_batch_size if actual_batch_size is not None else 0
		# print(f'seed is set to {seed} of type {type(seed)} placeholder is {self.seed_placeholder}')
		feed_dict[self.seed_placeholder] = seed
		return feed_dict

def run_tensors(self, x, y=None, tensors=None, seed=None, mode='test', save_dir='/', saver=None):
	"""
	Given inputs x, optional targets y, and optional dictionary of {id: tensor}, run the tensors (or all tensors
	in the graph) and return their values in the same manner as the tensors dictionary! (similar to how graphql
	works on a high level)
	"""
	start_time = time.time()
	time_lock_checks = 0
	feed_dict_time = 0
	load_vars_time = 0
	try_catch_time = 0
	time_session_run = 0
	outs_time = 0
	return_vals_time = 0
	get_time_info = False
	_clean_start = time.time()
	if self.cleanup:
		print("Attempting to run tensors when cleanup flag set, returning instead.")
		return {}
	time_lock_checks += time.time() - _clean_start
	_version_start = time.time()
	original_version = self.version
	time_lock_checks += time.time() - _version_start
	print("Starting run in {!s} mode...".format(mode))
	return_vals = dict()
	if isinstance(self.tf_graph, tf.Graph):
		if tensors is None:
			tensors = self.tensors
		_feed_start = time.time()
		feed_dict = self._construct_feed_dict(x=x, y=y, seed=seed, mode=mode)
		feed_dict_time = time.time() - _feed_start
		ids, outs = [], []
		_outs_start = time.time()
		for id, tensor in tensors.items():
			if isinstance(tensor, LobeTensor):
				tensor = tensor.value
			if isinstance(tensor, tf.Tensor) or isinstance(tensor, tf.Variable):
				ids.append(id)
				outs.append(tensor)
			else:
				return_vals[id] = tensor
		outs_time = time.time() - _outs_start
		if len(outs) > 0:
			_version_start = time.time()
			if self.version != original_version:
				raise GraphChanged("Graph changed during run_tensors execution.")
			time_lock_checks += time.time() - _version_start
			print("Running {!s} tensors from the graph!".format(len(outs)))
			_try_start = time.time()
			try:
				if get_time_info:
					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					_run_start = time.time()
					vals = self.session.run(
						outs,
						feed_dict=feed_dict,
						options=run_options,
						run_metadata=run_metadata
					)
					time_session_run = time.time() - _run_start
					# Create the Timeline object, and write it to a json
					tl = timeline.Timeline(run_metadata.step_stats)
					ctf = tl.generate_chrome_trace_format()
					with open('timeline.json', 'w') as f:
						f.write(ctf)
				else:
					_run_start = time.time()
					vals = self.session.run(
						outs,
						feed_dict=feed_dict,
					)
					time_session_run = time.time() - _run_start
			except tf.errors.FailedPreconditionError as e:
				try_catch_time = time.time() - _try_start
				_load_start = time.time()
				self.load_or_init_variables(save_dir=save_dir, saver=saver)
				load_vars_time = time.time() - _load_start
				_version_start = time.time()
				if self.version != original_version:
					raise GraphChanged("Graph changed during run_tensors execution.")
				time_lock_checks += time.time() - _version_start
				try:
					_run_start = time.time()
					vals = self.session.run(outs, feed_dict=feed_dict)
					time_session_run = time.time() - _run_start
				except tf.errors.InvalidArgumentError as e:
					_log.exception(e)
					raise Exception('Ran the graph without selected data')
			_return_start = time.time()
			for id, val in zip(ids, vals):
				return_vals[id] = val
			return_vals_time = time.time() - _return_start
	_version_start = time.time()
	if self.version != original_version:
		raise GraphChanged("Graph changed during run_tensors execution.")
	time_lock_checks += time.time() - _version_start
	print("Run took {!s}".format(make_time_units_string(time.time() - start_time)))
	return return_vals, [time_session_run, time_lock_checks, feed_dict_time, load_vars_time, try_catch_time, outs_time, return_vals_time]

# -----

app = Flask(__name__)

@app.route("/getPrediction")
def getPrediction():
	return "Prediction!"
