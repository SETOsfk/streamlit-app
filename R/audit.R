# R twin of python/audit.py — same questions, same folds (results/folds.csv).
#   Rscript R/audit.R        # base R only
root <- normalizePath(".")
for (k in 1:3) if (!file.exists(file.path(root, "Consumer_Shopping_Trends_2026.csv"))) root <- dirname(root)
d <- read.csv(file.path(root, "Consumer_Shopping_Trends_2026.csv"), stringsAsFactors = TRUE)
TARGET <- "shopping_preference"; CAT <- c("gender", "city_tier"); SPEND <- c("avg_store_spend", "avg_online_spend")
num <- setdiff(names(d), c(CAT, TARGET))
y <- d[[TARGET]]

# 1. independence of the behaviour columns
C <- cor(d[num], method = "spearman")
up <- abs(C[upper.tri(C)])
set.seed(20260927)
null_max <- replicate(200, max(abs(cor(sapply(d[num], sample), method = "spearman")[upper.tri(C)])))
# 2. uniform 0-10 scores
scores <- num[sapply(d[num], function(v) all(v %in% 0:10) && length(unique(v)) <= 11)]
uni_p <- sapply(scores, function(s) chisq.test(table(d[[s]]))$p.value)
# 3. separation (Kruskal-Wallis epsilon squared)
eps <- sort(sapply(num, function(f) unname(kruskal.test(d[[f]], y)$statistic)) / (nrow(d) - 1), decreasing = TRUE)
# 4. unpenalised multinomial logit (maximum likelihood, BFGS) on the Python folds
softmax <- function(E) { E <- E - apply(E, 1, max); exp(E) / rowSums(exp(E)) }
mlogit <- function(X, Y) {                      # X: n x p with intercept, Y: n x K indicators; class 1 = reference
  p <- ncol(X); K <- ncol(Y)
  B <- function(b) cbind(0, matrix(b, p, K - 1))
  nll <- function(b) { E <- X %*% B(b); m <- apply(E, 1, max); -sum(rowSums(Y * E) - m - log(rowSums(exp(E - m)))) }
  grad <- function(b) -as.vector(crossprod(X, (Y - softmax(X %*% B(b)))[, -1, drop = FALSE]))
  B(optim(numeric(p * (K - 1)), nll, grad, method = "BFGS", control = list(maxit = 5000, reltol = 1e-14))$par)
}
fold <- read.csv(file.path(root, "results", "folds.csv"))$fold
cv <- function(cols) {
  P <- matrix(0, nrow(d), nlevels(y), dimnames = list(NULL, levels(y)))
  for (k in sort(unique(fold))) {
    tr <- fold != k
    X <- d[cols]; isnum <- cols %in% num
    mu <- sapply(X[tr, isnum, drop = FALSE], mean); sdv <- sapply(X[tr, isnum, drop = FALSE], sd)
    X[isnum] <- sweep(sweep(X[isnum], 2, mu), 2, sdv, "/")
    MM <- model.matrix(~ ., X)                   # intercept + standardised numbers + dummy-coded categories
    Bk <- mlogit(MM[tr, , drop = FALSE], model.matrix(~ y - 1)[tr, ])
    P[!tr, ] <- softmax(MM[!tr, , drop = FALSE] %*% Bk)
  }
  pred <- factor(levels(y)[max.col(P, ties.method = "first")], levels(y))
  rec <- sapply(levels(y), function(g) mean(pred[y == g] == g))
  c(accuracy = mean(pred == y), balanced_accuracy = mean(rec),
    log_loss = -mean(log(pmax(P[cbind(seq_along(y), as.integer(y))], 1e-15))))
}
models <- rbind(behaviour_only = cv(c(setdiff(num, SPEND), CAT)), store_spend_only = cv("avg_store_spend"), all_columns = cv(c(num, CAT)))

cat(sprintf("max |rho| %.4f over %d pairs (%d above 0.1); null median %.4f\n", max(up), length(up), sum(up > 0.1), median(null_max)))
cat(sprintf("uniform 0-10 scores: %d of %d with p > 0.01\n", sum(uni_p > 0.01), length(uni_p)))
print(round(head(eps, 5), 4))
print(round(models, 4))

py <- file.path(root, "results", "audit.json")
if (file.exists(py)) {
  M <- jsonlite::fromJSON(py)
  pm <- do.call(rbind, lapply(M$models[rownames(models)], function(r) unlist(r[c("accuracy", "balanced_accuracy", "log_loss")])))
  cat("\nParity with Python (max abs difference):\n")
  print(signif(c(max_rho = abs(max(up) - M$independence$max_abs_rho),
                 epsilon2 = max(abs(eps[M$separation$feature] - M$separation$epsilon2)),
                 accuracy_behaviour_and_spend = max(abs(models[1:2, "accuracy"] - pm[1:2, "accuracy"])),
                 accuracy_all = abs(models[3, "accuracy"] - pm[3, "accuracy"])), 3))
  cat("All columns: the classes are perfectly separable, so the unpenalised likelihood has no finite optimum;\n",
      "R (BFGS) and scikit-learn (lbfgs) stop at different large coefficients: accuracy agrees, log loss does not.\n")
}
