import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from beartype import beartype

logger = logging.getLogger(__name__)


@beartype
@dataclass
class LithoData:
    """Class for preprocessing lithology data.

    Args:
        file_path (Path): Path to the file containing the lithology data.
        description_col (str): Column name containing the description of the lithology.
        primary_texture_col (str): Column name containing the primary texture of the lithology.
        secondary_texture_col (str): Column name containing the secondary texture of the lithology.
        primary_admixture_col (str): Column name containing the primary admixture of the lithology.
        secondary_admixture_col (str): Column name containing the secondary admixture of the lithology.
        color_col (str): Column name containing the color of the lithology.
        combine_labels (bool): Whether to combine the primary and secondary texture and admixture labels.
        x_coord_col (str): Column name containing the x-coordinates.
        y_coord_col (str): Column name containing the y-coordinates.
    """

    file_path: Path
    description_col: str
    primary_texture_col: str
    secondary_texture_col: str
    primary_admixture_col: str
    secondary_admixture_col: str
    color_col: str
    combine_labels: bool
    x_coord_col: str = "x"
    y_coord_col: str = "y"

    def __post_init__(self):
        """Read the data from the file_path."""
        self.df = pd.read_csv(self.file_path).dropna(subset=[self.description_col])
        self.filter_cols = [
            self.description_col,
            self.primary_texture_col,
            self.secondary_texture_col,
            self.primary_admixture_col,
            self.secondary_admixture_col,
            self.color_col,
        ]
        self.labels = [
            self.primary_texture_col,
            self.secondary_texture_col,
            self.primary_admixture_col,
            self.secondary_admixture_col,
        ]
        self.labels_extended = self.labels + [self.color_col]

    def _replace_idem_with_description(self):
        """The description in the data may contain the word `idem` indicating
        that some description is the same as the last given description. In such cases,
        we replace `idem` with the last known description. In some cases, suffixes are added to `idem`,
        we add them to the last available description. e add them to the last available description.
        Derived from https://doi.org/10.1016/j.acags.2025.100229
        """
        idem = self.df[self.description_col].str.lower().str.startswith("idem")
        if idem.sum() == 0:
            logger.log(logging.INFO, "No 'idem' found in the data.")
        else:
            logger.log(logging.INFO, f"Found {idem.sum()} 'idem' in the data.")
            replace_idx = [*map(self._find_previous_nonidem, idem[idem].index.tolist())]
            new_description = self.df.iloc[replace_idx][self.description_col].reset_index(drop=True)
            assert new_description.isna().sum() == 0, ValueError(
                "New description contains NaN values."
            )
            # in case of suffixes after idem, we need to add them to the description.
            idem_description = (
                self.df[idem][self.description_col]
                .str.partition("idem")
                .reset_index(drop=True)
                .iloc[:, -1]
            )
            assert idem_description.isna().sum() == 0, ValueError(
                "New description contains NaN values."
            )
            new_description = list(new_description + idem_description)
            assert len(new_description) == idem.sum(), ValueError(
                "Length of new description does not match the number of 'idem' found."
            )
            self.df.loc[idem, self.description_col] = new_description

    def _find_previous_nonidem(self, rownr: int) -> int:
        """Find the previous row that does not start with 'idem'.
        Args:
            rownr (int): Row number to start searching from.
        Returns:
            int: Row number of the previous row that does not start with 'idem'.
        """
        cur_row = rownr
        while self.df.iloc[cur_row][self.description_col].lower().startswith("idem"):
            cur_row -= 1

        # ensure we are considering rows with same coordinates
        assert all(
            self.df.iloc[rownr][[self.x_coord_col, self.y_coord_col]]
            == self.df.iloc[cur_row][[self.x_coord_col, self.y_coord_col]]
        ), ValueError("Coordinates do not match.")
        return cur_row

    def _remove_multiple_intervals_description(self):
        """Drop observations where the description describes different intervals in one entry.
        Derived from https://doi.org/10.1016/j.acags.2025.100229
        """
        interval_description = self.df[self.description_col].str.contains(
            "van [0-9]+ tot [0-9]+ .* van .* tot .*|(?:^|KERNSTROOK [0-9]*|Kern)[ ]?[0-9\.\,]+[ m]?(?:[- ]+|tot )[0-9\.\,]+[ ]?m",
            case=False,
            regex=True,
        )
        assert interval_description.isna().sum() == 0, ValueError(
            "Interval description contains NaN values."
        )
        logger.log(
            logging.INFO,
            f"Found {interval_description.sum()} descriptions in the data that contain intervals in a single entry.",
        )
        self.df = self.df.drop(index=self.df.loc[interval_description].index)

    def _remove_leading_spaces(self):
        """Remove leading spaces from the description."""
        self.df[self.description_col] = self.df[self.description_col].str.lstrip()

    def _filter_cols(self):
        """Filter the columns."""
        self.df = self.df.drop(
            columns=[col for col in self.df.columns if col not in self.filter_cols]
        )

    def _filter_rows(self):
        """Filter the rows."""
        self._drop_missing_labels()
        self._drop_low_frequency_labels()

    def _drop_missing_labels(self):
        """Drop rows with missing labels."""
        len_before = len(self.df)
        self.df = self.df.dropna(
            subset=self.labels,
            how="all",
        )
        len_after = len(self.df)
        logger.log(
            logging.INFO,
            f"Filtered {len_before - len_after} rows with missing values in primary and secondary texture and admixture.",
        )

        len_before = len(self.df)
        self.df = self.df.dropna(
            subset=self.color_col,
        )
        len_after = len(self.df)
        logger.log(
            logging.INFO,
            f"Filtered {len_before - len_after} rows with missing values in color.",
        )

    def _drop_low_frequency_labels(self):
        """Drop labels with low frequency based on primary texture."""
        freq = self.df[self.primary_texture_col].value_counts()
        low_freq_labels = freq[freq < 10].index
        # filter out low frequency labels
        self.df = self.df[~self.df[self.primary_texture_col].isin(low_freq_labels)]
        logger.log(
            logging.INFO,
            f"Filtered {len(low_freq_labels)} low frequency labels in primary texture.",
        )

        for label in [l for l in self.labels_extended if l != self.primary_texture_col]:
            freq = self.df[label].value_counts()
            low_freq_labels = freq[freq < 10].index
            # filter out low frequency labels
            self.df.loc[:, label] = self.df[label].apply(
                lambda x: "Other" if x in low_freq_labels else x
            )
            logger.log(
                logging.INFO,
                f"Replaced {len(low_freq_labels)} low frequency occurance of {label} with `Other`.",
            )

    def _combine_labels(self):
        """Create combined labels for the lithology."""
        self.df.loc[:, self.labels] = self.df[self.labels].fillna("None")
        self.df["combined_texture"] = self.df[self.primary_texture_col].str.cat(
            self.df[self.secondary_texture_col], sep="_"
        )
        self.df["combined_admixture"] = self.df[self.primary_admixture_col].str.cat(
            self.df[self.secondary_admixture_col], sep="_"
        )

    def preprocess_data(self):
        """Preprocess the text data."""
        self._remove_leading_spaces()
        self._replace_idem_with_description()
        self._remove_multiple_intervals_description()
        self._filter_cols()
        self._filter_rows()
        if self.combine_labels:
            self._combine_labels()
