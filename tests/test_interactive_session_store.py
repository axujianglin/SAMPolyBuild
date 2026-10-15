import unittest

from interactive_api.errors import InteractiveAPIError
from interactive_api.session_store import InteractionState, SessionStore


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = SessionStore(ttl_seconds=10, max_sessions=1, clock=self.clock)

    def _create(self):
        return self.store.create(
            image_id="image-id",
            file_name="tile.png",
            width=16,
            height=16,
            prepared_image={"embedding": 1},
        )

    def test_limit_delete_and_resource_release(self):
        entry = self._create()
        entry.interaction_states["building"] = InteractionState(
            "token", "building", 1, (1, 1, 10, 10), ((3, 3),), (),
            ((1, 1), (10, 1), (10, 10)), object(),
        )
        with self.assertRaises(InteractiveAPIError) as caught:
            self._create()
        self.assertEqual(caught.exception.code, "session_limit_reached")
        self.assertTrue(self.store.delete(entry.session_id))
        self.assertIsNone(entry.prepared_image)
        self.assertEqual(entry.interaction_states, {})
        self.assertFalse(self.store.delete(entry.session_id))

    def test_expired_session_uses_distinct_error(self):
        entry = self._create()
        self.clock.value = 111.0
        with self.assertRaises(InteractiveAPIError) as caught:
            with self.store.locked_session(entry.session_id):
                pass
        self.assertEqual(caught.exception.code, "session_expired")
        self.assertEqual(caught.exception.status_code, 410)
        self.assertIsNone(entry.prepared_image)

    def test_successful_touch_uses_sliding_idle_ttl(self):
        entry = self._create()
        self.clock.value = 105.0
        with self.store.locked_session(entry.session_id) as locked:
            self.store.touch(locked)
        self.clock.value = 112.0
        with self.store.locked_session(entry.session_id) as locked:
            self.assertEqual(locked.session_id, entry.session_id)


if __name__ == "__main__":
    unittest.main()
