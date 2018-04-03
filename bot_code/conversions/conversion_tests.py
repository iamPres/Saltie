

import tempfile

from bot_code.conversions.input.input_formatter import InputFormatter
from bot_code.conversions import output_formatter
from bot_code.conversions import binary_converter
from bot_code.conversions import transpose_converter
from game_data_struct import GameTickPacket
import struct

def test_create_input_array_is_idempodent():
    # pre-test: This used to fail due to mutated external state in conversion functions.
    game_tick_packet = GameTickPacket()
    input_formatter = InputFormatter(1,0)
    array1 = input_formatter.create_input_array(game_tick_packet, passed_time=0.0)
    array2 = input_formatter.create_input_array(game_tick_packet, passed_time=0.0)
    assert all(x==y for x,y in zip(array1, array2))


def write_test_data_to_file(replay_file):
    # prepare data
    team = 1
    player_index = 2
    bot_hash = 41
    is_eval = False
    game_tick_packet = GameTickPacket()
    game_tick_packet.gamecars[player_index].Team = 1
    game_tick_packet.gamecars[player_index].Boost = 30
    game_tick_packet.gamecars[player_index].Location.X = 31.
    input_formatter = InputFormatter(team, player_index)
    test_data_pairs = [
        (
            input_formatter.create_input_array(game_tick_packet, passed_time=42.0),
            [0.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            input_formatter.create_input_array(game_tick_packet, passed_time=52.0),
            [0.0, 53.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]

    # Note: This is API is fragile. Any mistake in ordering or ommission will throw things out of whack.

    # Write header
    binary_converter.write_header_to_file(replay_file, bot_hash, is_eval)

    # Write body
    for state_array, output_vector_array in test_data_pairs:
        binary_converter.write_array_to_file(replay_file, state_array)
        binary_converter.write_array_to_file(replay_file, output_vector_array)

def read_from_file_and_assert(replay_file):
    tuples = list(binary_converter.iterate_data(replay_file, verbose=False))
    # print([len(x) for x in tuples])
    for i, (state_array, output_vector_array, pair_number) in enumerate(tuples):
        state = output_formatter.get_advanced_state(state_array)
        # state = output_formatter.get_basic_state(state_array)

        # Test things that should be the same between the ticks
        # print (state_array)
        # assert state.car_info.Boost == 30
        # assert state.car_info.Location.X == 31.0

        # Test things that are different between the ticks
        if i == 0:
            # assert state.passed_time == 42.
            assert output_vector_array[1] == 43.
        elif i == 1:
            # assert state.passed_time == 51.
            assert output_vector_array[1] == 53.
        else:
            assert False, 'only expected two ticks worth of data'

        # print(state.car_info.Team)  # Debatable whether this should be 0 or 1 as we never asked for the field to be rotated.


def test_read_write_preserves_data():
    """
    Tests that the following chain produces the same data as originally provided:
    game_tick_packet
        input_formatter
            binary_converter
                .bin file
            binary_converter
        output_formatter
    tensorflow_object (lambda: None)
    """
    with tempfile.TemporaryFile() as replay_file:
        write_test_data_to_file(replay_file)
        replay_file.seek(0)
        read_from_file_and_assert(replay_file)

def test_as_non_overlapping_pairs():
    assert list(binary_converter.as_non_overlapping_pairs([1,2,3,4,5,6])) == [(1,2), (3,4), (5,6)]

    # Should iterate properly until the end and then raise an exception.
    try:
        for x,y in binary_converter.as_non_overlapping_pairs([1,2,3]):
            pass
        assert False, 'expected failure'
    except Exception:
        pass
    assert x == 1
    assert y == 2


def test_transpose_file():
    with tempfile.TemporaryFile() as original_file:
        original_file.write(bytes([3,4,5]))
        original_file.write(bytes([6,7,8]))

        original_file.seek(0)
        assert original_file.read(3) == bytes([3,4,5])
        assert original_file.read(3) == bytes([6,7,8])

        original_file.seek(0)
        with tempfile.TemporaryFile() as transposed:
            transpose_converter.transpose_file(original_file, transposed, 3)

            transposed.seek(0)
            assert transposed.read(2) == bytes([3,6])
            assert transposed.read(2) == bytes([4,7])
            assert transposed.read(2) == bytes([5,8])

            # Transpose back (note: requires different row length)
            transposed.seek(0)
            with tempfile.TemporaryFile() as back_again:
                transpose_converter.transpose_file(transposed, back_again, 2)
                original_file.seek(0)
                back_again.seek(0)
                assert original_file.read() == back_again.read()

def test_transpose_file_with_header():
    with tempfile.TemporaryFile() as original_file:
        original_file.write(bytes([1,2]))    # header
        original_file.write(bytes([3,4,5]))  # tick 1
        original_file.write(bytes([6,7,8]))  # tick 2

        original_file.seek(0)
        assert original_file.read(2) == bytes([1,2])
        assert original_file.read(3) == bytes([3,4,5])
        assert original_file.read(3) == bytes([6,7,8])

        original_file.seek(0)
        with tempfile.TemporaryFile() as transposed:
            transpose_converter.transpose_file_with_header(original_file, transposed, 2, 3)

            transposed.seek(0)
            assert transposed.read(2) == bytes([1,2])
            assert transposed.read(2) == bytes([3,6])
            assert transposed.read(2) == bytes([4,7])
            assert transposed.read(2) == bytes([5,8])

            # Transpose back (note: requires different row length)
            transposed.seek(0)
            with tempfile.TemporaryFile() as back_again:
                transpose_converter.transpose_file_with_header(transposed, back_again, 2, 2)
                original_file.seek(0)
                back_again.seek(0)
                assert original_file.read() == back_again.read()


test_create_input_array_is_idempodent()
test_read_write_preserves_data()
test_as_non_overlapping_pairs()
test_transpose_file()
test_transpose_file_with_header()
print (' === ALL TESTS PASSED === ')
