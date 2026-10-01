;;;; run.scm - run the nanograd-components test suite
;;;
;;; Runs each test file in its own csi process and reports one overall
;;; result.  Separate processes keep the files independent: they define
;;; top-level helpers with the same names but different values (such as
;;; the tolerance tol), they share module state such as the random seed,
;;; and each ends with (test-exit), which would stop a combined run.
;;;
;;; Usage:
;;;   csi -s tests/run.scm                   run every test file
;;;   csi -s tests/run.scm test-layer ...    run only the named files
;;;
;;; Arguments that do not name a test file are ignored; chicken-install
;;; -test passes the egg name and version.  The exit status is 0 when
;;; every file passes and 1 otherwise.

(import scheme
        (chicken base)
        (chicken format)
        (chicken pathname)
        (chicken process)
        (chicken process-context)
        (srfi 1))

;; Test files in the default run, in order.  test-am-models covers the
;; lazy (array-morphisms) models.
(define test-files
  '("test-gradcheck"
    "test-am-models"
    "test-mlp-mixer"
    "test-transformer"
    "test-resnet-numerical"
    "test-resnet"))

;; The directory holding this script, so that the suite can be run from
;; any working directory.
(define tests-directory
  (or (pathname-directory (program-name)) "."))

(define (test-file-path name)
  (make-pathname tests-directory name "scm"))

;; Runs one test file with the csi that runs this script and returns
;; its exit status.  Output goes straight to this process's output.
(define (run-test-file name)
  (printf "~%==== ~A ====~%" name)
  (flush-output)
  (let ((pid (process-run (executable-pathname)
                          (list "-s" (test-file-path name)))))
    (let-values (((pid normal-exit? status) (process-wait pid)))
      (if normal-exit? status 128))))

(define selected
  (let ((requested (filter (lambda (arg) (member arg test-files))
                           (command-line-arguments))))
    (if (null? requested) test-files requested)))

(define results
  (map (lambda (name) (cons name (run-test-file name))) selected))

(define failed
  (filter (lambda (r) (not (zero? (cdr r)))) results))

(printf "~%==== Summary ====~%")
(for-each (lambda (r)
            (printf "  ~A ~A~%"
                    (if (zero? (cdr r)) "pass" "FAIL")
                    (car r)))
          results)
(printf "~A of ~A test files passed.~%"
        (- (length results) (length failed))
        (length results))

(exit (if (null? failed) 0 1))
