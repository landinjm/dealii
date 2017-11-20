// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_scalapack_h
#define dealii_scalapack_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_SCALAPACK

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_support.h>
#include <deal.II/base/mpi.h>
#include <mpi.h>

#include <memory>

DEAL_II_NAMESPACE_OPEN


// Forward declaration of class ScaLAPACKMatrix for ProcessGrid
template <typename NumberType> class ScaLAPACKMatrix;


/**
 * A class taking care of setting up a two-dimensional processor grid.
 * For example an MPI communicator with 5 processes can be arranged into a
 * 2x2 grid with 5-th processor being inactive:
 * @code
 *      |   0     |   1
 * -----| ------- |-----
 * 0    |   P0    |  P1
 *      |         |
 * -----| ------- |-----
 * 1    |   P2    |  P3
 * @endcode
 *
 * A shared pointer to this class is provided to ScaLAPACKMatrix matrices to
 * perform block-cyclic distribution.
 *
 * Note that this class allows to setup a process grid which has fewer
 * MPI cores than the total number of cores in the communicator.
 *
 * @author Benjamin Brands, 2017
 */
class ProcessGrid
{
public:

  /**
   * Declare class ScaLAPACK as friend to provide access to private members, e.g. the MPI Communicator
   */
  template <typename NumberType> friend class ScaLAPACKMatrix;

  /**
   * Constructor for a process grid for a given @p mpi_communicator .
   * The pair @p grid_dimensions contains the user-defined numbers of process rows and columns.
   * Their product should be less or equal to the total number of cores
   * in the @p mpi_communicator.
   */
  ProcessGrid(MPI_Comm mpi_communicator,
              const std::pair<unsigned int,unsigned int> &grid_dimensions);

  /**
   * Constructor for a process grid for a given @p mpi_communicator.
   * In this case the process grid is heuristically chosen based on the
   * dimensions and block-cyclic distribution of a target matrix provided
   * in @p matrix_dimensions and @p block_sizes.
   *
   * The maximum number of MPI cores one can utilize is $\min\{\frac{M}{MB}\frac{N}{NB}, Np\}$, where $M,N$
   * are the matrix dimension and $MB,NB$ are the block sizes and $Np$ is the number of
   * processes in the @p mpi_communicator. This function then creates a 2D processor grid
   * assuming the ratio between number of process row $p$ and columns $q$ to be
   * equal the ratio between matrix dimensions $M$ and $N$.
   */
  ProcessGrid(MPI_Comm mpi_communicator,
              const std::pair<unsigned int,unsigned int> &matrix_dimensions,
              const std::pair<unsigned int,unsigned int> &block_sizes);

  /**
  * Destructor.
  */
  virtual ~ProcessGrid();

  /**
  * Return the number of rows in the processes grid.
  */
  unsigned int get_process_grid_rows() const;

  /**
  * Return the number of columns in the processes grid.
  */
  unsigned int get_process_grid_columns() const;

private:

  /**
   * Send @p count values stored consequently starting at @p value from
   * the process with rank zero to processes which
   * are not in the process grid.
   */
  template <typename NumberType>
  void send_to_inactive(NumberType *value, const int count=1) const;

  /**
  * An MPI communicator with all processes.
  */
  MPI_Comm mpi_communicator;

  /**
  * An MPI communicator with inactive processes and the process with rank zero.
  */
  MPI_Comm mpi_communicator_inactive_with_root;

  /**
  * BLACS context. This is equivalent to MPI communicators and is
  * used by ScaLAPACK.
  */
  int blacs_context;

  /**
  * Rank of this MPI process.
  */
  const unsigned int this_mpi_process;

  /**
  * Total number of MPI processes.
  */
  const unsigned int n_mpi_processes;

  /**
  * Number of rows in the processes grid.
  */
  int n_process_rows;

  /**
  * Number of columns in the processes grid.
  */
  int n_process_columns;

  /**
  * Row of this process in the grid.
  */
  int this_process_row;

  /**
  * Column of this process in the grid.
  */
  int this_process_column;

  /**
  * A flag which is true for processes within the 2D process grid.
  */
  bool active;

};


/**
 * A wrapper class around ScaLAPACK parallel dense linear algebra.
 *
 * ScaLAPACK assumes that matrices are distributed according to the
 * block-cyclic decomposition scheme. An $M$ by $N$ matrix is first decomposed
 * into $MB$ by $NB$ blocks which are then uniformly distributed across
 * the 2D process grid $p*q \le Np$.
 *
 * For example, a global real symmetric matrix of order 9 is stored in
 * upper storage mode with block sizes 4 × 4:
 * @code
 *                0                       1                2
 *     ┌                                                       ┐
 *     | -6.0  0.0  0.0  0.0  |   0.0 -2.0 -2.0  0.0  |   -2.0 |
 *     |   .  -6.0 -2.0  0.0  |  -2.0 -4.0  0.0 -4.0  |   -2.0 |
 * 0   |   .    .  -6.0 -2.0  |  -2.0  0.0  2.0  0.0  |    6.0 |
 *     |   .    .    .  -6.0  |   2.0  0.0  2.0  0.0  |    2.0 |
 *     | ---------------------|-----------------------|------- |
 *     |   .    .    .    .   |  -8.0 -4.0  0.0 -2.0  |    0.0 |
 *     |   .    .    .    .   |    .  -6.0  0.0 -4.0  |   -6.0 |
 * 1   |   .    .    .    .   |    .    .  -4.0  0.0  |    0.0 |
 *     |   .    .    .    .   |    .    .    .  -4.0  |   -4.0 |
 *     | ---------------------|-----------------------|------- |
 * 2   |   .    .    .    .   |    .    .    .    .   |  -16.0 |
 *     └                                                       ┘
 * @endcode
 * may be distributed using the 2x2 process grid:
 * @code
 *      |   0 2   |   1
 * -----| ------- |-----
 * 0    |   P00   |  P01
 * 2    |         |
 * -----| ------- |-----
 * 1    |   P10   |  P11
 * @endcode
 * with the following local arrays:
 * @code
 * p,q  |             0              |           1
 * -----|----------------------------|----------------------
 *      | -6.0  0.0  0.0  0.0  -2.0  |   0.0 -2.0 -2.0  0.0
 *      |   .  -6.0 -2.0  0.0  -2.0  |  -2.0 -4.0  0.0 -4.0
 *  0   |   .    .  -6.0 -2.0   6.0  |  -2.0  0.0  2.0  0.0
 *      |   .    .    .  -6.0   2.0  |   2.0  0.0  2.0  0.0
 *      |   .    .    .    .  -16.0  |    .    .    .    .
 * -----|----------------------------|----------------------
 *      |   .    .    .    .    0.0  |  -8.0 -4.0  0.0 -2.0
 *      |   .    .    .    .   -6.0  |    .  -6.0  0.0 -4.0
 *  1   |   .    .    .    .    0.0  |    .    .  -4.0  0.0
 *      |   .    .    .    .   -4.0  |    .    .    .  -4.0
 * @endcode
 *
 * The choice of the block size is a compromise between a sufficiently large
 * sizes for efficient local/serial BLAS, but one that is also small enough to achieve
 * good parallel load balance.
 *
 * Below we show a strong scaling example of ScaLAPACKMatrix::invert()
 * on up to 5 nodes each composed of two Intel Xeon 2660v2 IvyBridge sockets
 * 2.20GHz, 10 cores/socket. Calculations are performed on square processor
 * grids 1x1, 2x2, 3x3, 4x4, 5x5, 6x6, 7x7, 8x8, 9x9, 10x10.
 *
 * @image html scalapack_invert.png
 *
 * @ingroup Matrix1
 * @author Denis Davydov, Benjamin Brands, 2017
 */
template <typename NumberType>
class ScaLAPACKMatrix : protected TransposeTable<NumberType>
{
public:

  /**
   * Declare the type for container size.
   */
  typedef unsigned int size_type;

  /**
   * Constructor for a rectangular matrix with rows and columns provided in
   * @p sizes, and distributed using the grid @p process_grid.
   */
  ScaLAPACKMatrix(const std::pair<size_type,size_type> &sizes,
                  const std::shared_ptr<const ProcessGrid> process_grid,
                  const std::pair<size_type,size_type> &block_sizes = std::make_pair(32,32),
                  const LAPACKSupport::Property property = LAPACKSupport::Property::general);

  /**
   * Constructor for a square matrix of size @p size, and distributed
   * using the process grid in @p process_grid.
   */
  ScaLAPACKMatrix(const size_type size,
                  const std::shared_ptr<const ProcessGrid> process_grid,
                  const size_type block_size = 32,
                  const LAPACKSupport::Property property = LAPACKSupport::Property::symmetric);

  /**
   * Destructor
   */
  virtual ~ScaLAPACKMatrix();

  /**
   * Assign @p property to this matrix.
   */
  void set_property(const LAPACKSupport::Property property);

  /**
   * Assignment operator from a regular FullMatrix.
   *
   * @note This function should only be used for relatively small matrix
   * dimensions. It is primarily intended for debugging purposes.
   */
  ScaLAPACKMatrix<NumberType> &
  operator = (const FullMatrix<NumberType> &);

  /**
   * Copy the contents of the distributed matrix into @p matrix.
   *
   * @note This function should only be used for relatively small matrix
   * dimensions. It is primarily intended for debugging purposes.
   */
  void copy_to (FullMatrix<NumberType> &matrix) const;

  /**
   * Compute the Cholesky factorization of the matrix using ScaLAPACK
   * function <code>pXpotrf</code>. The result of factorization is stored in this object.
   */
  void compute_cholesky_factorization ();

  /**
   * Invert the matrix by first computing Cholesky factorization and then
   * building the actual inverse using <code>pXpotri</code>. The inverse is stored
   * in this object.
   */
  void invert();

  /**
   * Compute all eigenvalues of a real symmetric matrix using <code>pXsyev</code>.
   * If successful, the computed @p eigenvalues are arranged in ascending order.
   */
  void eigenvalues_symmetric (std::vector<NumberType> &eigenvalues);

  /**
   * Compute all eigenpairs of a real symmetric matrix using <code>pXsyev</code>.
   * If successful, the computed @p eigenvalues are arranged in ascending order.
   * The eigenvectors are stored in the columns of the matrix, thereby
   * overwriting the original content of the matrix.
   */
  void eigenpairs_symmetric (std::vector<NumberType> &eigenvalues);

  /**
   * Estimate the the condition number of a SPD matrix in the $l_1$-norm.
   * The matrix has to be in the Cholesky state (see compute_cholesky_factorization()).
   * The reciprocal of the
   * condition number is returned in order to avoid the possibility of
   * overflow when the condition number is very large.
   *
   * @p a_norm must contain the $l_1$-norm of the matrix prior to calling
   * Cholesky factorization.
   *
   * @note An alternative is to compute the inverse of the matrix
   * explicitly and manually constructor $k_1 = ||A||_1 ||A^{-1}||_1$.
   */
  NumberType reciprocal_condition_number(const NumberType a_norm) const;

  /**
   * Compute the $l_1$-norm of the matrix.
   */
  NumberType l1_norm() const;

  /**
   * Compute the $l_{\infty}$ norm of the matrix.
   */
  NumberType linfty_norm() const;

  /**
   * Compute the Frobenius norm of the matrix.
   */
  NumberType frobenius_norm() const;

  /**
   * Number of rows of the $M \times N$ matrix.
   */
  size_type m() const;

  /**
   * Number of columns of the $M \times N$ matrix.
   */
  size_type n() const;

private:

  /**
   * Number of local rows on this MPI processes.
   */
  int local_m() const;

  /**
   * Number of local columns on this MPI process.
   */
  int local_n() const;

  /**
   * Return the global row number for the given local row @p loc_row .
   */
  int global_row(const int loc_row) const;

  /**
   * Return the global column number for the given local column @p loc_column.
   */
  int global_column(const int loc_column) const;

  /**
   * Read access to local element.
   */
  NumberType local_el(const int loc_row, const int loc_column) const;

  /**
   * Write access to local element.
   */
  NumberType &local_el(const int loc_row, const int loc_column);

  /**
   * Calculate the norm of a distributed dense matrix using ScaLAPACK's
   * internal function.
   */
  NumberType norm(const char type) const;

  /**
   * Since ScaLAPACK operations notoriously change the meaning of the matrix
   * entries, we record the current state after the last operation here.
   */
  LAPACKSupport::State state;

  /**
   * Additional property of the matrix which may help to select more
   * efficient ScaLAPACK functions.
   */
  LAPACKSupport::Property property;

  /**
   * A shared pointer to a ProcessGrid object which contains a BLACS context
   * and a MPI communicator, as well as other necessary data structures.
   */
  std::shared_ptr<const ProcessGrid> grid;

  /**
   * Number of rows in the matrix.
   */
  int n_rows;

  /**
   * Number of columns in the matrix.
   */
  int n_columns;

  /**
   * Row block size.
   */
  int row_block_size;

  /**
   * Column block size.
   */
  int column_block_size;

  /**
   * Number of rows in the matrix owned by the current process.
   */
  int n_local_rows;

  /**
   * Number of columns in the matrix owned by the current process.
   */
  int n_local_columns;

  /**
   * ScaLAPACK description vector.
   */
  int descriptor[9];

  /**
   * Workspace array.
   */
  mutable std::vector<NumberType> work;

  /**
   * Integer workspace array.
   */
  mutable std::vector<int> iwork;

  /**
   * A character to define where elements are stored in case
   * ScaLAPACK operations support this.
   */
  const char uplo;

  /**
   * The process row of the process grid over which the first row
   * of the global matrix is distributed.
   */
  const int first_process_row;

  /**
   * The process column of the process grid over which the first column
   * of the global matrix is distributed.
   */
  const int first_process_column;

  /**
   * Global row index that determines where to start a submatrix.
   * Currently this equals unity, as we don't use submatrices.
   */
  const int submatrix_row;

  /**
   * Global column index that determines where to start a submatrix.
   * Currently this equals unity, as we don't use submatrices.
   */
  const int submatrix_column;

};

// ----------------------- inline functions ----------------------------

template <typename NumberType>
inline
NumberType
ScaLAPACKMatrix<NumberType>::local_el(const int loc_row, const int loc_column) const
{
  return (*this)(loc_row,loc_column);
}



template <typename NumberType>
inline
NumberType &
ScaLAPACKMatrix<NumberType>::local_el(const int loc_row, const int loc_column)
{
  return (*this)(loc_row,loc_column);
}

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_SCALAPACK

#endif
