# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""
NumPy Array Editor Dialog based on Qt
"""

# pylint: disable=C0103
# pylint: disable=R0903
# pylint: disable=R0911
# pylint: disable=R0201

# Standard library imports
import io

# Third party imports
from qtpy.compat import from_qvariant, to_qvariant
from qtpy.QtCore import (QAbstractTableModel, QItemSelection,
                         QItemSelectionRange, QModelIndex, Qt, Slot)

from qtpy.QtGui import QColor, QCursor, QDoubleValidator, QKeySequence

from qtpy.QtWidgets import (QAbstractItemDelegate, QApplication, QCheckBox,
                            QComboBox, QDialog, QDialogButtonBox, QGridLayout,
                            QHBoxLayout, QInputDialog, QItemDelegate, QLabel,
                            QLineEdit, QMenu, QMessageBox, QPushButton,
                            QSpinBox, QStackedWidget, QTableView, QVBoxLayout,
                            QWidget)

import numpy as np

# Local imports
from spyder.widgets.variableexplorer.arrayeditor import SUPPORTED_FORMATS, is_number, is_float, get_idx_rect, \
    LARGE_COLS, LARGE_NROWS, LARGE_SIZE, ArrayEditor, ArrayModel, ArrayDelegate, ArrayView


from spyder.config.base import _
from spyder.config.fonts import DEFAULT_SMALL_DELTA
from spyder.config.gui import get_font, config_shortcut
from spyder.py3compat import (io, is_binary_string, is_string,
                              is_text_string, PY3, to_binary_string,
                              to_text_string)
from spyder.utils import icon_manager as ima
from spyder.utils.qthelpers import add_actions, create_action, keybinding


# =============================================================================
# Main classes
# =============================================================================
class QuantityArrayModel(ArrayModel):
    """Array Editor Table Model"""

    # ROWS_TO_LOAD = 500
    # COLS_TO_LOAD = 40

    def __init__(self, data, format="%.3f", xlabels=None, ylabels=None,
                 readonly=False, parent=None):
        ArrayModel.__init__(self, data, format="%.3f", xlabels=None,
                            ylabels=None, readonly=False, parent=None)

        # data = quantity_data.m

        self.dialog = parent
        self.changes = {}
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.readonly = readonly
        self.test_array = np.array([0], dtype=data.dtype)

        # for complex numbers, shading will be based on absolute value
        # but for all other types it will be the real part
        if data.dtype in (np.complex64, np.complex128):
            self.color_func = np.abs
        else:
            self.color_func = np.real

        # Backgroundcolor settings
        huerange = [.66, .99]  # Hue
        self.sat = .7  # Saturation
        self.val = 1.  # Value
        self.alp = .6  # Alpha-channel

        self._data = data
        self._format = format

        self.total_rows = self._data.m.shape[0]
        self.total_cols = self._data.m.shape[1]
        size = self.total_rows * self.total_cols

        try:
            self.vmin = np.nanmin(self.color_func(data))
            self.vmax = np.nanmax(self.color_func(data))
            if self.vmax == self.vmin:
                self.vmin -= 1
            self.hue0 = huerange[0]
            self.dhue = huerange[1] - huerange[0]
            self.bgcolor_enabled = True
        except (TypeError, ValueError):
            self.vmin = None
            self.vmax = None
            self.hue0 = None
            self.dhue = None
            self.bgcolor_enabled = False

        # Use paging when the total size, number of rows or number of
        # columns is too large
        if size > LARGE_SIZE:
            self.rows_loaded = self.ROWS_TO_LOAD
            self.cols_loaded = self.COLS_TO_LOAD
        else:
            if self.total_rows > LARGE_NROWS:
                self.rows_loaded = self.ROWS_TO_LOAD
            else:
                self.rows_loaded = self.total_rows
            if self.total_cols > LARGE_COLS:
                self.cols_loaded = self.COLS_TO_LOAD
            else:
                self.cols_loaded = self.total_cols

    def get_value(self, index):
        i = index.row()
        j = index.column()
        return self.changes.get((i, j), self._data.m[i, j])

    def data(self, index, role=Qt.DisplayRole):
        """Cell content"""
        if not index.isValid():
            return to_qvariant()

        value = self.get_value(index)

        if is_binary_string(value):
            try:
                value = str(value)
            except:
                pass

        if role == Qt.DisplayRole:
            if value is np.ma.masked:
                return ''
            else:
                return to_qvariant(self._format % value)

        elif role == Qt.TextAlignmentRole:
            return to_qvariant(int(Qt.AlignCenter | Qt.AlignVCenter))

        elif role == Qt.BackgroundColorRole and self.bgcolor_enabled \
                and value is not np.ma.masked:
            hue = self.hue0 + self.dhue * (self.vmax - self.color_func(value)) / (self.vmax - self.vmin)
            hue = float(np.abs(hue))
            color = QColor.fromHsvF(hue, self.sat, self.val, self.alp)
            return to_qvariant(color)
        # elif role == Qt.FontRole:
        #     return to_qvariant(get_font(font_size_delta=DEFAULT_SMALL_DELTA))
        return to_qvariant()

    def setData(self, index, value, role=Qt.EditRole):
        """Cell content change"""
        if not index.isValid() or self.readonly:
            return False

        i = index.row()
        j = index.column()

        value = from_qvariant(value, str)
        dtype = self._data.m.dtype.name

        if dtype == "bool":
            try:
                val = bool(float(value))
            except ValueError:
                val = value.lower() == "true"
        elif dtype.startswith("string") or dtype.startswith("bytes"):
            val = to_binary_string(value, 'utf8')
        elif dtype.startswith("unicode") or dtype.startswith("str"):
            val = str(value)
        else:
            if value.lower().startswith('e') or value.lower().endswith('e'):
                return False
            try:
                val = complex(value)
                if not val.imag:
                    val = val.real
            except ValueError as e:
                QMessageBox.critical(self.dialog, "Error", "Value error: %s" % str(e))
                return False
        try:
            self.test_array[0] = val  # will raise an Exception eventually
        except OverflowError as e:
            print(type(e.message))
            QMessageBox.critical(self.dialog, "Error", "Overflow error: %s" % e.message)
            return False

        # Add change to self.changes
        self.changes[(i, j)] = val
        self.dataChanged.emit(index, index)
        if not is_string(val):
            if val > self.vmax:
                self.vmax = val
            if val < self.vmin:
                self.vmin = val
        return True


class QuantityArrayEditorWidget(QWidget):
    def __init__(self, parent, data, readonly=False, xlabels=None, ylabels=None):
        QWidget.__init__(self, parent)
        self.parent = parent
        self.data = data
        self.old_data_shape = None

        if len(self.data.shape) == 1:
            self.old_data_shape = self.data.shape
            self.data.shape = (self.data.shape[0], 1)

        elif len(self.data.shape) == 0:
            self.old_data_shape = self.data.shape
            self.data.shape = (1, 1)

        format = SUPPORTED_FORMATS.get(data.m.dtype.name, '%s')

        self.model = QuantityArrayModel(self.data,
                                        format=format,
                                        xlabels=xlabels,
                                        ylabels=ylabels,
                                        readonly=readonly,
                                        parent=self)

        self.view = ArrayView(self,
                              self.model,
                              data.m.dtype,
                              data.shape)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignLeft)
        btn = QPushButton("Format")

        # disable format button for int type
        btn.setEnabled(is_float(data.m.dtype))
        btn_layout.addWidget(btn)
        btn.clicked.connect(self.change_format)
        btn = QPushButton("Resize")
        btn_layout.addWidget(btn)
        btn.clicked.connect(self.view.resize_to_contents)
        bgcolor = QCheckBox('Background color')
        bgcolor.setChecked(self.model.bgcolor_enabled)
        bgcolor.setEnabled(self.model.bgcolor_enabled)
        bgcolor.stateChanged.connect(self.model.bgcolor)
        btn_layout.addWidget(bgcolor)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def accept_changes(self):
        """Accept changes"""

        for (i, j), value in list(self.model.changes.items()):
            self.data.m[i, j] = value

        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    def reject_changes(self):
        """Reject changes"""
        if self.old_data_shape is not None:
            self.data.m.shape = self.old_data_shape

    def change_format(self):
        """Change display format"""
        format, valid = QInputDialog.getText(self, _('Format'),
                                             _("Float formatting"),
                                             QLineEdit.Normal,
                                             self.model.get_format())
        if valid:
            format = str(format)
            try:
                format % 1.1
            except:
                QMessageBox.critical(self, _("Error"),
                                     _("Format (%s) is incorrect") % format)
                return
            self.model.set_format(format)


class QuantityArrayEditor(QDialog):
    """Array Editor Dialog"""

    def setup_and_check(self, data, title='', readonly=False,
                        xlabels=None, ylabels=None):
        """
        Setup ArrayEditor:
        return False if data is not supported, True otherwise
        """
        self.data = data
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.readonly = readonly

        self.data.flags.writeable = True
        is_record_array = data.m.dtype.names is not None
        is_masked_array = isinstance(data.m, np.ma.MaskedArray)

        if data.ndim > 3:
            self.error(_("Arrays with more than 3 dimensions are not "
                         "supported"))
            return False
        if xlabels is not None and len(xlabels) != self.data.shape[1]:
            self.error(_("The 'xlabels' argument length do no match array "
                         "column number"))
            return False
        if ylabels is not None and len(ylabels) != self.data.shape[0]:
            self.error(_("The 'ylabels' argument length do no match array row "
                         "number"))
        if not is_record_array:
            dtn = data.dtype.name
            if dtn not in SUPPORTED_FORMATS and not dtn.startswith('str') \
               and not dtn.startswith('unicode'):
                arr = _("%s arrays") % data.dtype.name
                self.error(_("%s are currently not supported") % arr)
                return False

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowIcon(ima.icon('arredit'))
        if title:
            title = str(title) + " - " + _("Pint NumPy array")
        else:
            title = _("Array editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)
        self.resize(600, 500)

        # Stack widget
        self.stack = QStackedWidget(self)
        if is_masked_array:
            self.stack.addWidget(QuantityArrayEditorWidget(self,
                                                           data,
                                                           readonly,
                                                           xlabels,
                                                           ylabels))
            self.stack.addWidget(QuantityArrayEditorWidget(self,
                                                           data.data,
                                                           readonly,
                                                           xlabels,
                                                           ylabels))
            self.stack.addWidget(QuantityArrayEditorWidget(self,
                                                           data.mask,
                                                           readonly,
                                                           xlabels,
                                                           ylabels))
        elif data.ndim == 3:
            pass
        else:
            self.stack.addWidget(QuantityArrayEditorWidget(self,
                                                           data,
                                                           readonly,
                                                           xlabels,
                                                           ylabels))

        self.arraywidget = self.stack.currentWidget()
        self.stack.currentChanged.connect(self.current_widget_changed)
        self.layout.addWidget(self.stack, 1, 0)

        self.unitLabel = QLabel()
        self.unitLabel.setText('{}: {}'.format(_('Current unit'),
                                               _(str(data.u))))
        self.dimensionLabel = QLabel()
        self.dimensionLabel.setText('{}: {}'.format(_('Dimension'),
                                                _(str(data.dimensionality))))
        self.layout.addWidget(self.dimensionLabel, 2, 0)
        self.layout.addWidget(self.unitLabel, 3, 0)


        # Buttons configuration
        btn_layout = QHBoxLayout()

        self.unitLbl = QLabel("Conver to:")
        self.unitCombo = QComboBox()
        self.add_units_to_combo()
        i = self.unitCombo.findData(self.data.u)
        self.unitCombo.setCurrentIndex(i)
        self.unitCombo.currentIndexChanged.connect(self.handle_unit_change)
        btn_layout.addWidget(self.unitLbl)
        btn_layout.addWidget(self.unitCombo)
        self.converted_widget = None


        if is_masked_array or data.m.ndim == 3:

            names = ['Masked data', 'Data', 'Mask']
            if data.m.ndim == 3:
                # QSpinBox
                self.index_spin = QSpinBox(self, keyboardTracking=False)
                self.index_spin.valueChanged.connect(self.change_active_widget)
                # QComboBox
                names = [str(i) for i in range(3)]
                ra_combo = QComboBox(self)
                ra_combo.addItems(names)
                ra_combo.currentIndexChanged.connect(self.current_dim_changed)
                # Adding the widgets to layout
                label = QLabel("Axis:")
                btn_layout.addWidget(label)
                btn_layout.addWidget(ra_combo)
                self.shape_label = QLabel()
                btn_layout.addWidget(self.shape_label)
                label = QLabel("Index:")
                btn_layout.addWidget(label)
                btn_layout.addWidget(self.index_spin)
                self.slicing_label = QLabel()
                btn_layout.addWidget(self.slicing_label)
                # set the widget to display when launched
                self.current_dim_changed(self.last_dim)
            else:
                ra_combo = QComboBox(self)
                ra_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
                ra_combo.addItems(names)
                btn_layout.addWidget(ra_combo)
            if is_masked_array:
                label = QLabel("<u>Warning</u>: changes are applied separately")
                label.setToolTip("For performance reasons, changes applied " \
                                 "to masked array won't be reflected in " \
                                 "array's data (and vice-versa).")
                btn_layout.addWidget(label)
            btn_layout.addStretch()

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        btn_layout.addWidget(bbox)
        self.layout.addLayout(btn_layout, 4, 0)
        self.setMinimumSize(400, 300)
        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)

        return True

    def current_widget_changed(self, index):
        self.arraywidget = self.stack.widget(index)

    def change_active_widget(self, index):
        """
        This is implemented for handling negative values in index for
        3d arrays, to give the same behavior as slicing
        """
        string_index = [':'] * 3
        string_index[self.last_dim] = '<font color=red>%i</font>'
        self.slicing_label.setText((r"Slicing: [" + ", ".join(string_index) + "]") % index)
        if index < 0:
            data_index = self.data.m.shape[self.last_dim] + index
        else:
            data_index = index
        slice_index = [slice(None)] * 3
        slice_index[self.last_dim] = data_index

        stack_index = self.dim_indexes[self.last_dim].get(data_index)
        if stack_index == None:
            stack_index = self.stack.count()
            self.stack.addWidget(QuantityArrayEditor(self, self.data.m[slice_index]))
            self.dim_indexes[self.last_dim][data_index] = stack_index
            self.stack.update()
        self.stack.setCurrentIndex(stack_index)

    def add_units_to_combo(self):
        units = list(sorted(self.data.compatible_units()))
        for unit in units:
            self.unitCombo.addItem(str(unit), unit)

    def handle_unit_change(self, i):

        if self.converted_widget is not None:
            self.stack.removeWidget(self.converted_widget)

        self.old_data_shape = self.stack.widget(0).old_data_shape

        unit = self.unitCombo.currentData()
        new_data = self.data.to(unit)

        if self.old_data_shape is not None:
            new_data.shape = self.old_data_shape

        self.converted_widget = QuantityArrayEditorWidget(self,
                                                       new_data,
                                                       self.readonly,
                                                       self.xlabels,
                                                       self.ylabels)



        self.stack.addWidget(self.converted_widget)
        self.stack.setCurrentWidget(self.converted_widget)

    def set_unit(self, new_unit):
        self.data.ito(new_unit)

    def accept2(self):

        self.accept()
        # self.accept()

    @Slot()
    def accept(self):
        """Reimplement Qt method"""

        if self.converted_widget is not None:
            if self.converted_widget.old_data_shape is not None:
                self.converted_widget.data.m.shape = self.converted_widget.old_data_shape

            self.data = self.converted_widget.data


        for index in range(self.stack.count()):
            self.stack.widget(index).accept_changes()

        QDialog.accept(self)

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data

    def error(self, message):
        """An error occured, closing the dialog box"""
        QMessageBox.critical(self, "Pint Array editor", message)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.reject()

    @Slot()
    def reject(self):
        """Reimplement Qt method"""
        if self.arraywidget is not None:
            for index in range(self.stack.count()):
                self.stack.widget(index).reject_changes()
        QDialog.reject(self)


class QuantityEditor(QLineEdit):
    def __init__(self, parent=None, option=None, index=None):
        QLineEdit.__init__(self, parent)
