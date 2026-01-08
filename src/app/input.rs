// input.rs

use rustc_hash::FxHashSet;
use winit::{event::*, keyboard::*};

pub struct InputManager {
    held_keys: FxHashSet<KeyCode>,
    pressed_keys: FxHashSet<KeyCode>,
    released_keys: FxHashSet<KeyCode>,

    mouse_buttons_held: FxHashSet<MouseButton>,
    mouse_buttons_pressed: FxHashSet<MouseButton>,
    mouse_buttons_released: FxHashSet<MouseButton>,

    mouse_delta: (f64, f64),
    scroll_delta: f32,
}

impl InputManager {
    pub fn new() -> Self {
        return InputManager {
            held_keys: FxHashSet::default(),
            pressed_keys: FxHashSet::default(),
            released_keys: FxHashSet::default(),

            mouse_buttons_held: FxHashSet::default(),
            mouse_buttons_pressed: FxHashSet::default(),
            mouse_buttons_released: FxHashSet::default(),

            mouse_delta: (0.0, 0.0),
            scroll_delta: 0.0,
        };
    }

    pub fn begin_frame(&mut self) {
        self.pressed_keys.clear();
        self.released_keys.clear();
        self.mouse_buttons_pressed.clear();
        self.mouse_buttons_released.clear();
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed => {
                    if !self.held_keys.contains(key) {
                        self.pressed_keys.insert(*key);
                    }
                    self.held_keys.insert(*key);
                }

                ElementState::Released => {
                    self.held_keys.remove(key);
                    self.released_keys.insert(*key);
                }
            },

            WindowEvent::MouseInput { button, state, .. } => match state {
                ElementState::Pressed => {
                    if !self.mouse_buttons_held.contains(button) {
                        self.mouse_buttons_pressed.insert(*button);
                    }
                    self.mouse_buttons_held.insert(*button);
                }
                ElementState::Released => {
                    self.mouse_buttons_held.remove(button);
                    self.mouse_buttons_released.insert(*button);
                }
            },

            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(_, y) => self.scroll_delta += y,
                MouseScrollDelta::PixelDelta(p) => self.scroll_delta += p.y as f32,
            },

            _ => {}
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_delta.0 = -delta.0;
            self.mouse_delta.1 = delta.1;
        }
    }

    // Query API
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.held_keys.contains(&key)
    }

    pub fn just_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    pub fn just_released(&self, key: KeyCode) -> bool {
        self.released_keys.contains(&key)
    }

    pub fn mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    pub fn scroll(&self) -> f32 {
        self.scroll_delta
    }
}
